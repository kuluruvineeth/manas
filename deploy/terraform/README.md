# Cluster provisioning

The manifests in `deploy/k8s/` assume a cluster that already has GPU nodes. This is that cluster.

```bash
terraform init
terraform apply -var project_id=YOUR_PROJECT
$(terraform output -raw get_credentials)
```

## What it builds, and why each piece is there

Two node pools, not one. A GKE cluster can never scale to zero nodes — system pods have to
live somewhere — so a cheap CPU pool holds them and the GPU pool is free to drop to zero.
An idle cluster then costs the price of one `e2-standard-2`, not of a GPU.

The GPU pool pins itself to a single zone. GPUs are zonal and often scarce, and a pool spread
across zones can end up only partly satisfiable. `location_policy = "ANY"` tells the autoscaler
to prefer unused reservations and lower the chance of a Spot node being reclaimed.

Drivers install themselves. Modern GKE installs the NVIDIA driver on GPU nodes automatically,
so the old `daemonset-preloaded.yaml` installer is only needed if you deliberately disable that,
run an older cluster, or need a pinned driver version. We set `gpu_driver_version` explicitly
rather than inherit a default that depends on cluster version.

You do not need to write a toleration. GKE taints GPU nodes `nvidia.com/gpu=present:NoSchedule`,
but it also runs the `ExtendedResourceToleration` admission controller, which adds the matching
toleration to any pod that requests `nvidia.com/gpu`. Asking for the resource is enough. One
sharp edge: GKE only applies that taint when the cluster has at least one non-GPU node pool —
another reason the system pool exists.

## Sharing one GPU

`gpu_shared_clients` defaults to 4, which puts four replicas on one physical L4 through
time-slicing. Giving a 64M model an entire L4 wastes almost all of it.

Time-slicing is the only sharing mode available here. MIG partitions a card into hardware
isolated instances, but it needs an A100/H100 class GPU — not an L4 or T4. MPS is the third
mode and suits cooperative batch jobs rather than independent servers.

The catch worth knowing: time-slicing does not enforce per-client memory limits. Four tenants
share the card's memory on the honour system, and one greedy pod can trigger an out-of-memory
error in its neighbours. For identical replicas of one small model that is fine.

## Cost

Prices for `us-central1`, checked 2026-09-19 — verify before you rely on them.

| shape | GPU | on-demand | spot |
|---|---|---|---|
| `g2-standard-4` | 1× L4 24GB | $0.71/hr | $0.42/hr |
| `n1-standard-4` + T4 | 1× T4 16GB | ~$0.54/hr | — |

T4 is cheaper per hour; L4 is a much newer architecture with more memory and usually wins on
cost per token. `use_spot` defaults to true, which roughly halves the bill in exchange for the
node being reclaimable at any moment — correct for a demo, wrong for anything serving users.

The GPU pool at zero nodes costs nothing, so the standing cost of an idle cluster is the system
pool plus the GKE management fee.

## Tearing it down

```bash
terraform destroy -var project_id=YOUR_PROJECT
```

A GPU node pool left running bills by the hour whether or not anything is using it. This is the
easiest way to lose real money on this project.
