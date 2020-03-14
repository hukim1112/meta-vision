import tensorflow as tf
import os


class Protonet_trainer():
    def __init__(self, model, train_step, train_ds, val_step, val_ds, config, optimizer):
        self.model = model
        self.train_step = train_step
        self.train_ds = train_ds
        self.val_step = val_step
        self.val_ds = val_ds
        self.config = config
        self.optimizer = optimizer
        self.train_loss = tf.metrics.Mean(name='train_loss')
        self.val_loss = tf.metrics.Mean(name='val_loss')
        self.train_acc = tf.metrics.Mean(name='train_accuracy')
        self.val_acc = tf.metrics.Mean(name='val_accuracy')

    def on_start_epoch(self):
        self.train_loss.reset_states()
        self.val_loss.reset_states()
        self.train_acc.reset_states()
        self.val_acc.reset_states()

    def train_episode(self, epi):
        if epi % 20 == 0:
            print(f"Episode {epi}")
        support, query = self.train_ds.get_next_episode()
        loss, pred, eq, acc = self.train_step(support, query, self.model, self.optimizer)
        self.train_loss(loss)
        self.train_acc(acc)

    def on_end_epoch(self, epoch):
        test_n_episode = self.config['test']['n_episode']
        # Validation
        for i_episode in range(test_n_episode):
            support, query = self.val_ds.get_next_episode()
            loss, pred, eq, acc = self.val_step(support, query, self.model)
            self.val_loss(loss)
            self.val_acc(acc)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, ' \
                   'Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch + 1, self.train_loss.result(), self.train_acc.result() * 100,
                              self.val_loss.result(), self.val_acc.result() * 100))
        if (epoch + 1) % 10 == 0:
            self.model.save(os.path.join(
                self.config['checkpoint_dir'], self.config['model_name'] + '_{}.h5'.format(epoch + 1)))

