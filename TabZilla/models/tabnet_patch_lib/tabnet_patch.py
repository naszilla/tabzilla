from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
import numpy as np
from sklearn.metrics import log_loss

class TabNetClassifierPatched(TabNetClassifier):
    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        #output_dim, train_labels = infer_output_dim(y_train)
        output_dim, train_labels = infer_output_dim(np.array(range(self.num_classes)))
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)


    def _predict_epoch(self, name, loader):
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        self.network.eval()

        list_y_true = []
        list_y_score = []

        # Main loop
        for batch_idx, (X, y) in enumerate(loader):
            scores = self._predict_batch(X)
            list_y_true.append(y)
            list_y_score.append(scores)

        y_true, scores = self.stack_batches(list_y_true, list_y_score)

        #metrics_logs = self._metric_container_dict[name](y_true, scores)
        metrics_logs = {f"{name}_logloss": log_loss(y_true, scores, labels=range(self.num_classes))}
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return

