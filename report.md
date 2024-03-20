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