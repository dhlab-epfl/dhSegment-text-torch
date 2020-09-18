# dhSegment-text-torch

This repository contains an add-on to [dhSegment torch](https://github.com/dhlab-epfl/dhSegment-torch) to use it with text embeddings maps.

For more details about text embeddings map, please see the following publication:


```
Barman, Raphaël, Ehrmann, Maud, Clematide, Simon, Ares Oliveira, Sofia, and Kaplan, Frédéric  (2020).
Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers.
Journal of Data Mining and Digital Humanities. https://arxiv.org/abs/2002.06144
```

## Usage

This repository introduces new input code for using text embeddings maps as well as new networks.

Using dhSegment torch training script, the following config parameters must be changed:
- `train_dataset` and `val_dataset` should now be of type `image_text_csv` (note that patches datasets are not supported).
- `train_loader` and `val_loader` must be set to `text_data_loader`.
- `model` type should be set to `text_segmentation_model`.
- Either the `encoder` or `decoder` should be set to the `text_` variant (currently supported architectures are `text_resnet50` and `text_unet`).
- The `text_` `encoder` or `decoder` should have the following additional parameters:
  -  `"embeddings_encoder": {"target_embeddings_size": 300}` set to the size of the embeddings (here 300).
  - `"embeddings_level": 0` set to the level in the network where the embeddings map should be input (here 0)
  
An example config file can be found in `example_conf.json`.
  
In addition to these changes to the config file, the training script should be modified to by adding `import dh_segment_text_torch` to the top.
