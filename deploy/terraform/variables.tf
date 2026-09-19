variable "project_id" {
  type        = string
  description = "GCP project to create the cluster in."
}

variable "region" {
  type        = string
  description = "Region for the cluster control plane."
  default     = "us-central1"
}

variable "gpu_zone" {
  type        = string
  description = "Single zone for the GPU pool. GPUs are zonal and scarce, so pin one."
  default     = "us-central1-a"
}

variable "cluster_name" {
  type    = string
  default = "manas"
}

variable "system_machine_type" {
  type        = string
  description = "Cheap CPU node that holds system pods so the GPU pool can reach zero."
  default     = "e2-standard-2"
}

variable "gpu_machine_type" {
  type        = string
  description = "The L4 is bundled into the g2 family and cannot be attached to other types."
  default     = "g2-standard-4"
}

variable "gpu_type" {
  type    = string
  default = "nvidia-l4"
}

variable "gpu_max_nodes" {
  type    = number
  default = 2
}

variable "use_spot" {
  type        = bool
  description = "Spot is roughly 40% cheaper but can be reclaimed at any moment."
  default     = true
}

variable "gpu_shared_clients" {
  type        = number
  description = <<-EOT
    Replicas that share one physical GPU via time-slicing. A 64M model has no business
    holding a whole L4. Set to 1 to disable sharing. GKE caps this at 48 and does not
    enforce per-client memory limits, so every tenant must stay within its own budget.
    Time-slicing is the only option here: MIG needs an A100/H100 class card, not an L4.
  EOT
  default     = 4
}
