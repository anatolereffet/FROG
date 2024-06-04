def train(dataloader, model, loss_fn, metrics_fn, optimizer, epoch):
    """Train the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
        epoch: an integer, the current epoch number.
    """
    model.train()
    for batch_idx, (X, y, gender, filename) in tqdm(
            enumerate(training_generator), total=len(training_generator)):
        X, y = X.to(device), y.to(device)
        y = torch.reshape(y, (len(y), 1))
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            step = batch // 100 * (epoch + 1)
            mlflow.log_metric("loss", f"{loss:2f}", step=step)
            mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
            print(
                f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")


def evaluate(dataloader, model, loss_fn, metrics_fn, epoch):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        epoch: an integer, the current epoch number.
    """
    num_batches = len(dataloader)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    results_list = []
    with torch.no_grad():
        for batch_idx, (X, y, gender, filename) in tqdm(
                enumerate(validation_generator), total=len(validation_generator)):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            eval_accuracy += metrics_fn(pred, y)
            for i in range(len(X)):
                results_list.append(
                    {"pred": float(y_pred[i]), "target": float(
                        y[i]), "gender": float(gender[i])}
                )

    results_df = pd.DataFrame(results_list)
    results_male = results_df.loc[results_df["gender"] > 0.5]
    results_female = results_df.loc[results_df["gender"] < 0.5]

    eval_loss /= num_batches
    eval_accuracy /= num_batches
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch)
    mlflow.log_metric("results_male", f"{results_male:2f}", step=epoch)
    mlflow.log_metric("results_female", f"{results_female:2f}", step=epoch)

    print(
        f"Eval metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n", f"results_male: {results_male}", f"results_female: {results_female}")
