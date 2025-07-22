# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example comparing training a neural network with the NTK dynamics.

In this example, we train a neural network on a small subset of MNIST using an
MSE loss and SGD. We compare this training with the analytic function space
prediction using the NTK. Data is loaded using tensorflow datasets.
"""

from absl import app
from jax import grad
from jax import jit
from jax import random
from jax.example_libraries import optimizers
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util
import numpy as np
from flaxmodels import VGG19
import jax
import jax.image
from flax.training import train_state
import optax
import jax.random as jrandom
import jax.nn as jnn


_LEARNING_RATE = 1.0  # Learning rate to use during training.
_TRAIN_SIZE = 1000  # Increased for better VGG-19 training.
_TEST_SIZE = 1000  # Increased for better VGG-19 evaluation.
_TRAIN_TIME = 1000.0  # Continuous time denoting duration of training.
_BATCH_SIZE = 0


# Utility to preprocess MNIST images for VGG/ViT

def preprocess_for_vgg_vit(images):
    # images: (N, 28, 28)
    # Convert to (N, 28, 28, 1)
    images = np.expand_dims(images, -1)
    # Repeat channels to get (N, 28, 28, 3)
    images = np.repeat(images, 3, axis=-1)
    # Resize to (N, 224, 224, 3)
    images = jax.image.resize(images, (images.shape[0], 224, 224, 3), method="bilinear")
    # Normalize to [0, 1]
    images = images / 255.0
    return images


def main(unused_argv):
    print("[DEBUG] Entered main function.")
    # Build data pipelines.
    print('Loading data.')
    x_train, y_train, x_test, y_test = datasets.get_dataset('mnist', _TRAIN_SIZE, _TEST_SIZE)

    # Preprocess for VGG/ViT
    x_train_vgg = preprocess_for_vgg_vit(np.array(x_train).reshape(-1, 28, 28))
    x_test_vgg = preprocess_for_vgg_vit(np.array(x_test).reshape(-1, 28, 28))

    print('[DEBUG] About to call train_vgg19...')
    # NTK experiment code removed to avoid OOM. Only VGG-19 will be trained and evaluated.

    # VGG-19 evaluation
    print('\nEvaluating VGG-19 on MNIST...')
    vgg = VGG19(num_classes=10, pretrained=None)
    rng = jax.random.PRNGKey(0)
    variables = vgg.init(rng, x_train_vgg[:1], train=False)
    # Forward pass for train and test
    logits_train = vgg.apply(variables, x_train_vgg, train=False)
    logits_test = vgg.apply(variables, x_test_vgg, train=False)
    # Print summary for train
    print('VGG-19 TRAIN RESULTS:')
    util.print_summary('VGG-19 train', y_train, logits_train, None, loss)
    # Print summary for test
    print('VGG-19 TEST RESULTS:')
    util.print_summary('VGG-19 test', y_test, logits_test, None, loss)

    # Data augmentation: random shift
    def random_shift(images, key, max_shift=3):
        # images: (N, 224, 224, 3)
        # Randomly shift images by up to max_shift pixels in x and y
        def shift(img, k):
            dx = jrandom.randint(k, (), -max_shift, max_shift+1)
            dy = jrandom.randint(k, (), -max_shift, max_shift+1)
            return jax.image.shift(img, (dx, dy, 0), mode='wrap')
        keys = jrandom.split(key, images.shape[0])
        return jax.vmap(shift)(images, keys)

    # Cross-entropy loss
    def cross_entropy_loss(logits, labels):
        one_hot = jnn.one_hot(labels, num_classes=10)
        return -jnp.mean(jnp.sum(one_hot * jnn.log_softmax(logits), axis=-1))

    def compute_metrics(logits, labels):
        loss = cross_entropy_loss(logits, labels)
        acc = jnp.mean(jnp.argmax(logits, -1) == labels)
        return {'loss': loss, 'accuracy': acc}

    # Training loop for VGG-19

    def train_vgg19(x_train, y_train, x_test, y_test, num_epochs=5, batch_size=4):
        print("[DEBUG] Entered train_vgg19 function.")
        print("[DEBUG] About to create VGG19 model instance.")
        vgg = VGG19(num_classes=10, pretrained=None)
        rng = jrandom.PRNGKey(0)
        print("[DEBUG] About to call vgg.init...")
        try:
            variables = vgg.init(rng, x_train[:1], train=True)
        except Exception as e:
            print(f"[ERROR] Exception during vgg.init: {e}")
            import traceback; traceback.print_exc()
            return
        print("[DEBUG] vgg.init completed successfully.")
        params = variables['params']
        tx = optax.adam(1e-3)
        state = train_state.TrainState.create(apply_fn=vgg.apply, params=params, tx=tx)
        n_train = x_train.shape[0]
        steps_per_epoch = n_train // batch_size
        try:
            for epoch in range(num_epochs):
                print(f"[DEBUG] Starting epoch {epoch+1}/{num_epochs}")
                # Shuffle
                perm = np.random.permutation(n_train)
                x_train_shuf = x_train[perm]
                y_train_shuf = y_train[perm]
                for i in range(steps_per_epoch):
                    batch_x = x_train_shuf[i*batch_size:(i+1)*batch_size]
                    batch_y = np.argmax(y_train_shuf[i*batch_size:(i+1)*batch_size], axis=-1)
                    # Data augmentation
                    aug_key = jrandom.fold_in(rng, epoch*steps_per_epoch + i)
                    batch_x = random_shift(batch_x, aug_key)
                    def loss_fn(params):
                        logits = vgg.apply({'params': params}, batch_x, train=True, rngs={'dropout': aug_key})
                        return cross_entropy_loss(logits, batch_y)
                    grads = jax.grad(loss_fn)(state.params)
                    state = state.apply_gradients(grads=grads)
                    # Print batch progress
                    if i % 10 == 0 or i == steps_per_epoch - 1:
                        logits = vgg.apply({'params': state.params}, batch_x, train=False)
                        batch_acc = np.mean(np.argmax(np.array(logits), -1) == np.array(batch_y))
                        batch_loss = float(cross_entropy_loss(logits, batch_y))
                        print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{steps_per_epoch} - Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}")
                print(f"Epoch {epoch+1}/{num_epochs} done.")
        except Exception as e:
            print(f"[ERROR] Exception during VGG-19 training: {e}")
            import traceback; traceback.print_exc()
        # Evaluate
        print("[DEBUG] Starting VGG-19 evaluation.")
        train_logits = vgg.apply({'params': state.params}, x_train, train=False)
        test_logits = vgg.apply({'params': state.params}, x_test, train=False)
        train_labels = np.argmax(y_train, axis=-1)
        test_labels = np.argmax(y_test, axis=-1)
        train_metrics = compute_metrics(train_logits, train_labels)
        test_metrics = compute_metrics(test_logits, test_labels)
        print("\nVGG-19 TRAIN RESULTS:")
        print(f"Accuracy: {train_metrics['accuracy']:.4f}, Loss: {train_metrics['loss']:.4f}")
        print("VGG-19 TEST RESULTS:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}, Loss: {test_metrics['loss']:.4f}")

    # VGG-19 training and evaluation
    print('\nTraining VGG-19 on MNIST with data augmentation and dropout...')
    train_vgg19(x_train_vgg, y_train, x_test_vgg, y_test, num_epochs=20, batch_size=32)


if __name__ == '__main__':
  app.run(main)
