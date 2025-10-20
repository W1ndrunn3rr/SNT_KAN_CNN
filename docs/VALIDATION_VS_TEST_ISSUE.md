# Analiza problemu: Validation 98% vs Test 35%

## 🔍 Problem

Model osiąga **98% accuracy na zbiorze walidacyjnym**, ale tylko **35% na zbiorze testowym**, pomimo że dane pochodzą z tego samego źródła.

## 🎯 Przyczyny

### 1. **Nieprawidłowy podział danych (GŁÓWNY PROBLEM)**

**Poprzednia implementacja:**
```python
# ❌ ZŁE - brak ustawionego seed
train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
```

**Problemy:**
- Brak seed → za każdym razem inny podział
- Brak reprodukowalności
- Model może "widzieć" różne dane walidacyjne przy różnych uruchomieniach

### 2. **Brak augmentacji danych**

**Poprzednio:**
- Te same transformacje dla train i validation
- Model nie uczy się robustnych cech
- Overfitting na konkretne przykłady

### 3. **Brak różnicowania train/val transforms**

**Co to oznacza:**
- Zbiór walidacyjny pochodzi z tych samych rozkładów co treningowy
- Model "zna" statystyki danych treningowych
- Test set może mieć inny rozkład (distribution shift)

## ✅ Rozwiązania

### 1. **Poprawiony DataProcessor**

```python
# ✅ DOBRE - z seedem i augmentacją
generator = torch.Generator().manual_seed(seed)
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=generator
)
```

**Zalety:**
- Reprodukowalny podział (seed=42)
- Różne transformacje dla train (z augmentacją) i val (bez augmentacji)
- Lepsze parametry DataLoader (pin_memory, persistent_workers)

### 2. **Augmentacja danych dla treningu**

```python
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomVerticalFlip(p=0.5),
transforms.RandomRotation(degrees=15),
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
```

**Dlaczego to pomaga:**
- Model widzi różne wersje tych samych obrazów
- Uczy się bardziej ogólnych cech
- Redukuje overfitting

### 3. **Konfigurowalność przez Hydra**

```yaml
data_params:
  val_split: 0.2  # 20% danych na walidację
  seed: 42        # Reprodukowalność
```

## 📊 Jak zdiagnozować problem

### Uruchom analizę danych:

```bash
just analyze-data
```

**Skrypt sprawdzi:**
- ✓ Rozkład klas w train vs test
- ✓ Statystyki obrazów (mean, std, rozmiary)
- ✓ Data leakage (duplikaty między zbiorami)
- ✓ Distribution shift

### Rezultaty analizy:

```
📊 Analyzing Training Set: data/train
Total samples: 25600
Classes: ['agri', 'barrenland', 'grassland', 'urban']
Class distribution:
  agri        :   6400 (25.00%)
  barrenland  :   6400 (25.00%)
  grassland   :   6400 (25.00%)
  urban       :   6400 (25.00%)

📊 Analyzing Test Set: data/test
Total samples: 6400
Classes: ['agri', 'barrenland', 'grassland', 'urban']
Class distribution:
  agri        :   1600 (25.00%)
  barrenland  :   1600 (25.00%)
  grassland   :   1600 (25.00%)
  urban       :   1600 (25.00%)
```

## 🎯 Co zrobić dalej?

### 1. Retrenuj model z nowymi ustawieniami

```bash
# Wyczyść stare logi
just clear-logs

# Trenuj z poprawionymi ustawieniami
just train KAN_FAST

# Monitoruj w TensorBoard
just tensorboard
```

### 2. Obserwuj metryki podczas treningu

**Oczekiwane zachowanie:**
- Validation accuracy powinna być niższa niż poprzednio (~85-90%)
- Test accuracy powinna być zbliżona do validation accuracy (±5%)
- Jeśli nadal duża różnica → możliwy distribution shift

### 3. Eksperymenty z regularizacją

Jeśli problem persystuje, spróbuj:

```bash
# Zwiększ augmentację
# Zmniejsz learning rate
uv run train model_params.learning_rate=5e-5

# Zmniejsz liczbę epok (early stopping)
uv run train model_params.max_epochs=20

# Zmień optimizer
uv run train model_params.optimizer=adamw
```

### 4. Porównaj różne modele

```bash
# Test z innymi architekturami
just train resnet50
just train efficientnet_b0
just train vit_b_16

# Zobacz który generalizuje najlepiej
just test-last resnet50
just test-last efficientnet_b0
```

## 📈 Oczekiwane wyniki po poprawkach

| Metryka | Przed | Po poprawkach | Cel |
|---------|-------|---------------|-----|
| Train Accuracy | ~99% | ~95-98% | ✓ Mniej overfitting |
| Val Accuracy | ~98% | ~85-92% | ✓ Bardziej realistyczne |
| Test Accuracy | ~35% | ~80-90% | ✓ Dobra generalizacja |
| Val/Test gap | 63% | <10% | ✓ Konsystencja |

## 🔧 Dodatkowe techniki

### 1. Early Stopping

Dodaj do train.py:

```python
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    verbose=True,
)
```

### 2. Learning Rate Scheduler

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
        },
    }
```

### 3. Mixup / CutMix augmentacja

Dla bardzo trudnych przypadków, rozważ zaawansowane augmentacje.

## 📝 Podsumowanie

**Problem był spowodowany przez:**
1. ✗ Brak seed w random_split → niereprodukowalny podział
2. ✗ Brak augmentacji → overfitting
3. ✗ Model "znał" rozkład danych treningowych zbyt dobrze

**Rozwiązanie:**
1. ✓ Fixed seed dla reprodukowalności
2. ✓ Augmentacja danych dla treningu
3. ✓ Osobne transformacje dla train/val
4. ✓ Konfigurowalność przez Hydra
5. ✓ Narzędzia do analizy (analyze_data.py)

**Następne kroki:**
1. Uruchom `just analyze-data` aby zdiagnozować dane
2. Retrenuj model z nowymi ustawieniami
3. Porównaj validation i test accuracy
4. Dostosuj hyperparametry jeśli potrzeba
