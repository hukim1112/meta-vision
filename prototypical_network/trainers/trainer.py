import tensorflow as tf
import os


class Protonet_trainer():
    def __init__(self, model, train_step, val_step, config, optimizer):
        self.model = model
        self.train_step = train_step
        self.val_step = val_step
        self.config = config
        self.optimizer = optimizer
        self.train_loss = tf.metrics.Mean(name='train_loss')
        self.val_loss = tf.metrics.Mean(name='val_loss')
        self.train_acc = tf.metrics.Mean(name='train_accuracy')
        self.val_acc = tf.metrics.Mean(name='val_accuracy')

    @staticmethod
    def on_start_epoch(self):
        self.train_loss.reset_states()
        self.val_loss.reset_states()
        self.train_acc.reset_states()
        self.val_acc.reset_states()

    @staticmethod
    def on_end_epoch(self, epoch):
        template = 'Epoch {}, Loss: {}, Accuracy: {}, ' \
                   'Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch + 1, self.train_loss.result(), self.train_acc.result() * 100,
                              self.val_loss.result(), self.val_acc.result() * 100))
        if (epoch + 1) != 0 & (epoch + 1) % 10 == 0:
            self.model.save(os.path.join(
                self.config['checkpoint_dir'], self.config['model_name'] + '_{}.h5'.format(epoch + 1)))

    @staticmethod
    def on_start_episode(self, support, query, epi):
        if epi % 20 == 0:
            print(f"Episode {epi}")
        loss, pred, eq, acc = self.train_step(
            support, query, self.model, self.optimizer)
        self.train_loss(loss)
        self.train_acc(acc)

    @staticmethod
    def on_end_episode(self, val_ds):
        test_n_episode = self.config['test']['n_episode']
        # Validation
        for i_episode in range(test_n_episode):
            support, query = val_ds.get_next_episode()
            loss, pred, eq, acc = self.val_step(support, query, self.model)
            self.val_loss(loss)
            self.val_acc(acc)