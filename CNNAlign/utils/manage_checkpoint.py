import os


class Saver():
    def __init__(self, ckpt_dir, save_type='latest', interval=None, max_to_keep=None):
        assert save_type in ['latest', 'best', 'local_minimum']
        if save_type == 'local_minimum':
            assert interval is not None
            self.interval = interval
            self.interval_queue = []
        self.ckpt_dir = ckpt_dir
        self.makedirs(self.ckpt_dir)
        self.save_type = save_type
        self.max_to_keep = max_to_keep
        self.eval_queue = []

    def makedirs(self, path):
        os.makedirs(path, exist_ok=True)

    def _queue(self, queue, epoch, loss, queue_size):
        queue.append({'epoch': epoch, 'loss': loss})
        if queue_size is not None:
            if len(queue) > queue_size:
                item = queue.pop(0)
                return item['epoch']
        return None

    def save_or_not(self, model, epoch, loss):
        epoch_to_delete = None
        if self.save_type == 'latest':
            self.save(model, epoch)
            epoch_to_delete = self._queue(
                self.eval_queue, epoch, loss, self.max_to_keep)
        elif self.save_type == 'best':
            if len(self.eval_queue) == 0:
                self.save(model, epoch)
                epoch_to_delete = self._queue(
                    self.eval_queue, epoch, loss, self.max_to_keep)
            elif self.eval_queue[-1]['loss'] > loss:
                self.save(model, epoch)
                epoch_to_delete = self._queue(
                    self.eval_queue, epoch, loss, self.max_to_keep)
        else:  # self.save_type == 'local_minimum'
              # x* is a local minimum when loss(x*) < loss(x), all x*-interval <= x <= x*+interval
            self._queue(self.interval_queue, epoch, loss, self.interval + 1)
            # this epoch is a minimum in interval.
            if min([x['loss'] for x in self.interval_queue]) == loss:
                self.save(model, epoch)
                self.minimum = epoch
                for i in [item['epoch'] for item in self.interval_queue[:-2]]:
                    self.delete(model, i)  # if there exists
            elif self.interval_queue[0]['epoch'] == self.minimum:
                # then the first model in interval_queue is a local minimum
                item = self.interval_queue[0]
                # epoch&loss of local minimum
                lm_epoch, lm_loss = item['epoch'], item['loss']
                epoch_to_delete = self._queue(
                    self.eval_queue, lm_epoch, lm_loss, self.max_to_keep)
        if epoch_to_delete is not None:
            self.delete(model, epoch_to_delete)
        print(self.eval_queue)
    def save(self, model, epoch):
        model.save(os.path.join(self.ckpt_dir,
                                "{}-{}.h5".format(model.model_name, epoch)))

    def delete(self, model, epoch):
        path = os.path.join(
            self.ckpt_dir, "{}-{}.h5".format(model.model_name, epoch))
        if os.path.isfile(path):
            os.remove(path)
        return

class dummy_model():
    def __init__(self):
        self.model_name = 'dummy'
    def save(self, path):
        pass
    def load(self, path):
        pass

def main():
    ckpt ="sss"
    saver = Saver(ckpt, save_type='local_minimum', interval=1, max_to_keep=3)
    model = dummy_model()

    val_loss = [10, 9, 8, 100, 20, 30, 7, 6, 50, 4, 30, 2, 1]
    for i, val in enumerate(val_loss):
        saver.save_or_not(model, i+1, val)


if __name__ == "__main__":
    main()