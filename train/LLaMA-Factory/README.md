# LlamaFactory Modified Version
This repository contains modified files based on [LlamaFactory](https://github.com/hiyouga/LlamaFactory) (Apache 2.0 License), focusing on streaming preprocessed dataset and load extracted feature.

## Key Modifications
| File Path | Modification Description |
|-----------|--------------------------|
| llamafactory/data/loader.py | Added support for  preprocessed dataset|
| llamafactory/data/mm_plugin.py | Added feature for loading extracted feature |

## How to Use
1. Clone the original LlamaFactory repository:
   ```bash
   git clone https://github.com/hiyouga/LlamaFactory.git
   cd LlamaFactory
   ```
2. Replace the corresponding files in the original repository with the modified files in this repo.
3. Follow the original LlamaFactory documentation to run the code.