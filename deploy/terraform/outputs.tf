output "cluster_name" {
  value = google_container_cluster.manas.name
}

output "cluster_location" {
  value = google_container_cluster.manas.location
}

output "get_credentials" {
  description = "Run this before kubectl apply."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.manas.name} --region ${google_container_cluster.manas.location} --project ${var.project_id}"
}
