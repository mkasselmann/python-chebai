from pathlib import Path
from types import SimpleNamespace

import lightning
import pandas as pd
import pytest
import torch

from chebai.models.electra import Electra
from chebai.models.ffn import FFN
from chebai.preprocessing.datasets.chebi import ChEBIOver50Char, ChEBIOver50SPE
from chebai.result.prediction import Predictor


class PredictionModel:
    def predict_step(self, batch, batch_idx):
        return {"preds": batch.y.float(), "labels": batch.y}


@pytest.fixture(params=[ChEBIOver50Char, ChEBIOver50SPE])
def evaluation_run(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("C\n", encoding="utf-8")
    codes = tmp_path / "codes.txt"
    codes.write_text("#version: 0.2\n", encoding="utf-8")
    reader_kwargs = {"token_path": str(tokens)}
    if request.param is ChEBIOver50SPE:
        reader_kwargs["codes_path"] = str(codes)
    dm = request.param(reader_kwargs=reader_kwargs, batch_size=2)
    run = Path(dm.processed_dir_main).resolve()
    (run / "classes.txt").write_text("1\n2\n", encoding="utf-8")
    pd.DataFrame(
        {"id": ["a", "b", "c"], "split": ["train", "validation", "test"]}
    ).to_csv(run / "splits.csv", index=False)
    torch.save(
        [
            {"ident": "a", "features": [1.0], "labels": [True, False]},
            {"ident": "b", "features": [2.0], "labels": [False, True]},
            {"ident": "c", "features": [3.0], "labels": [True, True]},
        ],
        Path(dm.processed_dir) / "data.pt",
    )
    predictor = Predictor.__new__(Predictor)
    predictor._dm = dm
    predictor._model = PredictionModel()
    predictor._classification_labels = ["1", "2"]
    predictor._evaluation_data = None
    return predictor, run


def test_original_splits_and_predictions(evaluation_run):
    predictor, run = evaluation_run
    predictions, labels = predictor.evaluate_predictions(processed_run_dir=run)
    assert predictions.tolist() == [[1.0, 1.0]]
    assert labels.tolist() == [[True, True]]

    predictions, labels = predictor.evaluate_predictions(
        partition="validation", processed_run_dir=run
    )
    assert predictions.tolist() == [[0.0, 1.0]]
    assert labels.tolist() == [[False, True]]


def test_checkpoint_records_training_run(evaluation_run):
    predictor, run = evaluation_run
    model = FFN(input_dim=1, out_dim=2, hidden_layers=[])
    model._trainer = SimpleNamespace(datamodule=predictor._dm)
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    assert checkpoint["evaluation_data"] == {
        "processed_run_dir": str(run),
        "splits_file_path": str(run / "splits.csv"),
        "splits_in_run": True,
    }
    predictor._evaluation_data = checkpoint["evaluation_data"]
    predictions, _ = predictor.evaluate_predictions()
    assert predictions.shape == (1, 2)


def test_real_checkpoint_load_and_evaluation(evaluation_run, tmp_path):
    predictor, run = evaluation_run
    dm = predictor._dm
    model = FFN(
        input_dim=1,
        out_dim=2,
        hidden_layers=[],
        classes_txt_file_path=str(run / "classes.txt"),
    )
    model._trainer = SimpleNamespace(datamodule=dm)
    model_hparams = dict(model.hparams)
    model_hparams.update(
        _class_path="chebai.models.ffn.FFN",
        _instantiator="lightning.pytorch.cli.instantiate_module",
    )
    dm_hparams = dict(dm.hparams)
    dm_hparams["_class_path"] = f"{type(dm).__module__}.{type(dm).__name__}"
    checkpoint = {
        "state_dict": model.state_dict(),
        "pytorch-lightning_version": lightning.__version__,
        "hyper_parameters": model_hparams,
        "datamodule_hyper_parameters": dm_hparams,
    }
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)

    loaded = Predictor(path, compile_model=False)
    predictions, labels = loaded.evaluate_predictions()
    assert predictions.shape == (1, 2)
    assert labels.tolist() == [[1, 1]]


def test_lightning_training_checkpoint(evaluation_run, tmp_path, monkeypatch):
    predictor, run = evaluation_run
    dm = predictor._dm
    dm.splits_file_path = str(run / "splits.csv")
    dm.hparams["_class_path"] = f"{type(dm).__module__}.{type(dm).__name__}"
    monkeypatch.setattr(dm, "prepare_data", lambda: None)
    monkeypatch.setattr(dm, "setup", lambda stage=None: None)
    model = FFN(
        input_dim=1,
        out_dim=2,
        hidden_layers=[],
        classes_txt_file_path=str(run / "classes.txt"),
        criterion=torch.nn.BCEWithLogitsLoss(),
        pass_loss_kwargs=False,
    )
    model.hparams["_class_path"] = "chebai.models.ffn.FFN"
    model.hparams["_instantiator"] = "lightning.pytorch.cli.instantiate_module"
    trainer = lightning.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
    )
    trainer.fit(model, datamodule=dm)
    path = tmp_path / "trained.ckpt"
    trainer.save_checkpoint(path)
    loaded = Predictor(path, compile_model=False)
    predictions, labels = loaded.evaluate_predictions()
    assert predictions.shape == (1, 2)
    assert labels.tolist() == [[1, 1]]


def test_electra_checkpoint_predictions(evaluation_run, tmp_path):
    predictor, run = evaluation_run
    dm = predictor._dm
    data_path = run / dm.reader.name() / "data.pt"
    records = torch.load(data_path, weights_only=False)
    for record in records:
        record["features"] = [int(value) for value in record["features"]]
    torch.save(records, data_path)
    model = Electra(
        input_dim=1,
        out_dim=2,
        classes_txt_file_path=str(run / "classes.txt"),
        config={
            "vocab_size": 16,
            "embedding_size": 8,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "max_position_embeddings": 16,
        },
    )
    model._trainer = SimpleNamespace(datamodule=dm)
    model_hparams = dict(model.hparams)
    model_hparams.update(
        _class_path="chebai.models.electra.Electra",
        _instantiator="lightning.pytorch.cli.instantiate_module",
    )
    dm_hparams = dict(dm.hparams)
    dm_hparams["_class_path"] = f"{type(dm).__module__}.{type(dm).__name__}"
    checkpoint = {
        "state_dict": model.state_dict(),
        "pytorch-lightning_version": lightning.__version__,
        "hyper_parameters": model_hparams,
        "datamodule_hyper_parameters": dm_hparams,
    }
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "electra.ckpt"
    torch.save(checkpoint, path)
    loaded = Predictor(path, compile_model=False)
    predictions, labels = loaded.evaluate_predictions()
    assert predictions.shape == (1, 2)
    assert labels.tolist() == [[1, 1]]
    assert torch.all((predictions >= 0) & (predictions <= 1))


def test_legacy_checkpoint_requires_original_run(evaluation_run):
    predictor, _ = evaluation_run
    with pytest.raises(ValueError, match="does not record"):
        predictor.evaluation_dataloader()


def test_moved_run_and_external_split(evaluation_run, tmp_path):
    predictor, run = evaluation_run
    external = tmp_path / "external.csv"
    external.write_text((run / "splits.csv").read_text(), encoding="utf-8")
    predictor._evaluation_data = {
        "processed_run_dir": "/remote/training/run",
        "splits_file_path": "/remote/external.csv",
        "splits_in_run": False,
    }
    with pytest.raises(ValueError, match="external splits"):
        predictor.evaluation_dataloader(processed_run_dir=run)
    predictions, _ = predictor.evaluate_predictions(
        processed_run_dir=run, splits_file_path=external
    )
    assert predictions.shape == (1, 2)


def test_rejects_mismatched_classes_and_missing_ids(evaluation_run):
    predictor, run = evaluation_run
    (run / "classes.txt").write_text("2\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Class labels"):
        predictor.evaluation_dataloader(processed_run_dir=run)

    (run / "classes.txt").write_text("1\n2\n", encoding="utf-8")
    (run / "splits.csv").write_text("id,split\nmissing,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="IDs.*missing"):
        predictor.evaluation_dataloader(processed_run_dir=run)


def test_missing_data_and_split_fail_explicitly(evaluation_run):
    predictor, run = evaluation_run
    with pytest.raises(FileNotFoundError, match="does not exist"):
        predictor.evaluation_dataloader(
            processed_run_dir=run, splits_file_path=run / "missing.csv"
        )
    (run / predictor._dm.reader.name() / "data.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Processed data not found"):
        predictor.evaluation_dataloader(processed_run_dir=run)
