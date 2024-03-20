# UIB INF265 Project 1
Group: **Project 1 18**  
Students:
- **Mats Omland Dyrøy (mdy020)**
- **Linus Krystad Raaen (zec018)**

---

- [Design](#design)
- [Models](#models)
  - [CnnV1](#cnnv1)
- [Variants](#variants)

## Design
We tried preprocessing the data with erosion and dilation to get rid of some of the noise, but the results werent great so we decided against using it for the final project. We also noticed some of the labels didnt match up with what we assumed to humbers to be and some of the boxes werent on top of where we would place the number so that could be a source of error when it comes to accuracy, however we didnt find a good fix against this so for now it will just be left as is.

## Models

### CnnV1

## Variants

### Global parameters:
| Parameter | Value |
| --------- | ----- |
| Batch size | `256` |
| Epoch count | `30` |
| Loss function | `CrossEntropyLoss` |
| Random seed | `420` |

### Variants
| No | Network | Learning rate | Momentum | Weight decay |
|:-: | ------- | -------- | ------- | --------- |
|  1 | `CnnV1` | $ 0.01 $ | $ 0.0 $ | $ 0.000 $ |
|  2 | `CnnV1` | $ 0.10 $ | $ 0.0 $ | $ 0.000 $ |
|  3 | `CnnV1` | $ 0.01 $ | $ 0.0 $ | $ 0.010 $ | 
|  4 | `CnnV1` | $ 0.01 $ | $ 0.9 $ | $ 0.000 $ | 
|  5 | `CnnV1` | $ 0.01 $ | $ 0.9 $ | $ 0.010 $ | 
|  6 | `CnnV1` | $ 0.01 $ | $ 0.9 $ | $ 0.001 $ | 
|  7 | `CnnV1` | $ 0.01 $ | $ 0.8 $ | $ 0.010 $ | 
|  8 | `CnnV1` | $ 0.10 $ | $ 0.9 $ | $ 0.010 $ | 
|  9 | `CnnV1` | $ 0.10 $ | $ 0.9 $ | $ 0.001 $ | 
| 10 | `CnnV1` | $ 0.10 $ | $ 0.9 $ | $ 0.010 $ | 
| 11 | `CnnV1` | $ 0.90 $ | $ 0.9 $ | $ 0.010 $ | 