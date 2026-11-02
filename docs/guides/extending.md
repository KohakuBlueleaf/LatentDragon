# Extending

Every model part follows the slot contract in [model](../concepts/model.md): it declares the
slots it reads and writes, and maps a slot dict to a slot dict. `examples/` shows parts
registered from outside the framework for paired data and for packed latents.

| Add | Do |
| --- | --- |
| Encoder, decoder | subclass `Part` (or `TensorPart` for one slot in, one out), set `inputs` and `outputs`, register in `ENCODER` or `DECODER` |
| Head | subclass `Part`, set `objective`; declare `last_layer` for an adaptive GAN weight; register in `HEAD` |
| Latent def | subclass `LatentDef` (or `TensorLatent`), implement `serialize`, `deserialize`, `pack`; register in `LATENT` |
| Norm, block | register in `NORM`, `BLOCK`; a block sets `spatial_reach` and `temporal_reach` for exact tiling |
| Decode strategy | subclass `DecodeStrategy`, list decoder methods in `requires`, register in `DECODE` |
| Quantizer codebook | register in `CODEBOOK`, return `(R, P, N, D)` |
| Critic | subclass `Critic`, set `tracks_stats`, register in `CRITIC` |
| GAN scheme | return a `GanScheme` from a factory in `GAN_SCHEME` |
| Latent view or term | subclass, register, list in `LATENT_VIEWS` or `LATENT_TERMS` |
| Domain | return a `Domain` from a factory in `DOMAIN`, naming its slots |
| Data type outside signals | write a new `Task` |

A dotted path works anywhere a registry name does, so experiments need no core edit.

The trainer reads only slot dicts and declared capabilities, so a new domain, encoder, latent or
decoder needs no edit outside its own files. A critic reads the data slot named by its `slot`.
