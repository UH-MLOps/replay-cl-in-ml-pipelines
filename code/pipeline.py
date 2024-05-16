import pandas as pd
import numpy as np
import torch
import lightning as L

from model import ShipPerformanceModel
from exemplar_set import ExemplarSet
from selector import get_random_selector
from dataset import get_train_loader, get_val_loader

GENERATIONS = 5
SEED = 42
BATCH_SIZE = 64

full_sensor_data = pd.read_csv('path/to/full_dataset.csv')

# split the dataset into generations and further split into train and val sets
train_sets = []
val_sets = []
for generation in np.array_split(full_sensor_data, GENERATIONS):
    train, val = np.array_split(generation, [int(0.8 * len(generation))])
    train_sets.append(train)
    val_sets.append(val)

# define experiments and their exemplar set sizes
EXPERIMENTS = {
  'replay': 3500,
  'all_data': len(full_sensor_data),
  'no_data': 0
}

for experiment_name, exemplar_set_size in EXPERIMENTS.items():
  # reset seeds for experiment run
  reset_seeds()

  model = ShipPerformanceModel(loss_fn=torch.nn.MSELoss(), 
                               optimizer_config={'optimizer': torch.optim.Adam, 'lr': 0.005})

  # init exemplar set
  exemplar_set = ExemplarSet(size=exemplar_set_size, 
                             selector=get_random_selector(seed=SEED), 
                             mode='ratios')

  # load train and validation data
  train, val = train_sets[0], val_sets[0]
  train_loader = get_train_loader(train, BATCH_SIZE)
  val_loader = get_val_loader(val)

  # train and eval base model
  trainer = L.Trainer(max_epochs=80)
  trainer.fit(model, train_loader, val_loader)
  eval(model, val_loader)

  for i in range(1, GENERATIONS):
    # update exemplar set with last generation data
    exemplar_set.sample_exemplars(train_sets[i - 1])

    # load current generation data and exemplar set
    train, val = train_sets[i], val_sets[i]
    exemplars = exemplar_set.get_exemplar_set()

    # combine and shuffle train and exemplar set
    train_set = pd.concat([train, exemplars]).sample(frac=1, random_state=SEED)

    # retrain and eval model on new generation data
    trainer = L.Trainer(max_epochs=30)
    trainer.fit(model, get_train_loader(train_set, BATCH_SIZE), get_val_loader(val))
    eval(model, val_loader)
