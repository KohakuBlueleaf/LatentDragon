# Extending

| Add | Do |
| --- | --- |
| Norm, block, encoder, trunk | register in `NORM`, `BLOCK`, `ENCODER`, `TRUNK` |
| Head | register in `HEAD`, pair in `HEAD_OBJECTIVES`, expose `anchor` |
| Bottleneck | subclass `Bottleneck`, register in `BOTTLENECK` |
| Quantizer codebook | register in `CODEBOOK`, return `(R, P, N, D)` |
| Critic | subclass `Critic`, set `tracks_stats`, register in `CRITIC` |
| GAN scheme | return a `GanScheme` from a factory in `GAN_SCHEME` |
| Latent view or term | subclass, register, list in `LATENT_VIEWS` or `LATENT_TERMS` |
| Domain | add a frozen record to `DOMAIN` |
| Data type outside signals | write a new `Task` |

A dotted path works anywhere a registry name does, so experiments need no core edit.

New domains add blocks, sources, critics and objectives. `training/`, `vq/` and
`bottleneck/` assume nothing about images, and there is deliberately no generic N-dimensional
conv layer.
