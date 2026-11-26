# Zero-shot evaluation: nomic-embed-v1.5

- Backend: burn-vulkan
- Manifest: `../data/neon/manifest.csv`, 700 images
- Model load 0.92 s; per image (decode + preprocess + embed) median 62 ms, p95 64 ms

## All settings, best macro accuracy first

| Framing | Prompts | Wording | Top-1 | Top-3 | Macro |
|---|---|---|---|---|---|
| centre-crop | photo | both | 74.0% | 92.6% | 74.0% |
| centre-crop | ensemble | both | 72.6% | 93.0% | 72.6% |
| centre-crop | specimen | both | 72.3% | 92.7% | 72.3% |
| centre-crop | bare | both | 71.6% | 93.3% | 71.6% |
| centre-crop | bare | common | 70.7% | 90.9% | 70.7% |
| centre-crop | specimen | common | 70.3% | 90.7% | 70.3% |
| centre-crop | photo | common | 70.1% | 90.9% | 70.1% |
| centre-crop | ensemble | common | 70.0% | 91.3% | 70.0% |
| pad | specimen | common | 67.3% | 89.4% | 67.3% |
| pad | photo | both | 67.0% | 90.9% | 67.0% |
| pad | photo | common | 66.7% | 88.1% | 66.7% |
| pad | specimen | both | 66.6% | 91.3% | 66.6% |
| pad | ensemble | common | 65.6% | 88.6% | 65.6% |
| pad | ensemble | both | 65.4% | 90.3% | 65.4% |
| pad | bare | common | 65.3% | 88.3% | 65.3% |
| pad | bare | both | 64.9% | 91.1% | 64.9% |
| centre-crop | ensemble | taxon | 55.0% | 88.0% | 55.0% |
| centre-crop | specimen | taxon | 54.7% | 83.9% | 54.7% |
| centre-crop | photo | taxon | 54.4% | 87.9% | 54.4% |
| pad | ensemble | taxon | 51.6% | 77.4% | 51.6% |
| pad | specimen | taxon | 51.4% | 73.6% | 51.4% |
| pad | photo | taxon | 51.3% | 75.1% | 51.3% |
| centre-crop | bare | taxon | 45.1% | 84.9% | 45.1% |
| pad | bare | taxon | 40.1% | 68.3% | 40.1% |

## Default setting (centre-crop, ensemble, common): centre-crop / ensemble / common

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 71.0% |
| mouse_or_vole | 100 | 97.0% |
| frog_or_toad | 100 | 12.0% |
| salamander | 100 | 98.0% |
| lizard | 100 | 21.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 91.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 71 | 29 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 0 | 97 | 0 | 1 | 0 | 0 | 2 |
| **frog_or_toad** | 1 | 0 | 12 | 74 | 0 | 9 | 4 |
| **salamander** | 0 | 0 | 0 | 98 | 1 | 0 | 1 |
| **lizard** | 0 | 0 | 0 | 72 | 21 | 0 | 7 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 6 | 0 | 3 | 91 |

## Best setting: centre-crop / photo / both

| Class | Images | Accuracy |
|---|---|---|
| shrew | 100 | 62.0% |
| mouse_or_vole | 100 | 97.0% |
| frog_or_toad | 100 | 15.0% |
| salamander | 100 | 92.0% |
| lizard | 100 | 60.0% |
| ground_beetle | 100 | 100.0% |
| mosquito | 100 | 92.0% |

Confusion matrix (rows = truth, columns = predicted):

| | shrew | mouse_or_vole | frog_or_toad | salamander | lizard | ground_beetle | mosquito |
|---|---|---|---|---|---|---|---|
| **shrew** | 62 | 38 | 0 | 0 | 0 | 0 | 0 |
| **mouse_or_vole** | 0 | 97 | 0 | 1 | 0 | 0 | 2 |
| **frog_or_toad** | 1 | 0 | 15 | 62 | 5 | 14 | 3 |
| **salamander** | 0 | 0 | 0 | 92 | 7 | 0 | 1 |
| **lizard** | 0 | 0 | 0 | 33 | 60 | 0 | 7 |
| **ground_beetle** | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **mosquito** | 0 | 0 | 0 | 5 | 0 | 3 | 92 |
