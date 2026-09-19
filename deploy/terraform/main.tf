terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source = "hashicorp/google"
      # guest_accelerator switched from list-assignment to block syntax in 6.0.0.
      version = "~> 8.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "manas" {
  name     = var.cluster_name
  location = var.region

  # The default pool is replaced immediately by the two pools below, which is the
  # documented way to manage node pools as separate resources.
  remove_default_node_pool = true
  initial_node_count       = 1

  deletion_protection = false

  monitoring_config {
    # DCGM ships GPU metrics to Managed Prometheus. Needed by anything that autoscales
    # or alerts on GPU behaviour; on 1.32.1-gke.1357000+ it is on by default anyway.
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }
}

# A GKE cluster never scales to zero nodes — system pods have to live somewhere. Keeping
# them on a cheap CPU pool is what lets the GPU pool drop to zero and cost nothing idle.
resource "google_container_node_pool" "system" {
  name     = "system"
  cluster  = google_container_cluster.manas.name
  location = google_container_cluster.manas.location

  autoscaling {
    min_node_count = 1
    max_node_count = 2
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = var.system_machine_type
    disk_size_gb = 50
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}

resource "google_container_node_pool" "gpu" {
  name     = "gpu"
  cluster  = google_container_cluster.manas.name
  location = google_container_cluster.manas.location

  # GPUs are zonal and scarce. Pinning to one zone avoids a multi-zone pool that can
  # only satisfy part of its capacity.
  node_locations = [var.gpu_zone]

  autoscaling {
    # Zero is the whole point: an idle cluster bills for the system pool only.
    min_node_count = 0
    max_node_count = var.gpu_max_nodes
    # ANY prioritises unused reservations and lowers preemption risk on Spot.
    location_policy = "ANY"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    # The L4 is bundled into the g2 machine type — it cannot be attached to anything else.
    machine_type = var.gpu_machine_type
    image_type   = "cos_containerd"
    disk_size_gb = 100
    spot         = var.use_spot
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = var.gpu_type
      count = 1

      gpu_driver_installation_config {
        # Modern GKE installs drivers itself; state it rather than inherit a version-dependent default.
        gpu_driver_version = "LATEST"
      }

      dynamic "gpu_sharing_config" {
        for_each = var.gpu_shared_clients > 1 ? [1] : []
        content {
          # A 64M model does not need a whole L4. Time-sharing puts several replicas on one
          # card; note GKE does not enforce per-client memory limits, so the pods must behave.
          gpu_sharing_strategy       = "TIME_SHARING"
          max_shared_clients_per_gpu = var.gpu_shared_clients
        }
      }
    }
  }
}
