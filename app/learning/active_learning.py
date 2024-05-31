from typing import Dict, Optional, List

from app.constants import DEFAULT_UNLABELED_LABEL
from app.learning.dataset import Dataset
from app.learning.models import NERModel
from app.data_types import Sentence, Annotation, Word, AssistantConf, LabelData


class ActiveLearningManager:
    """
    A class to manage the active learning process.

    This class is responsible for providing sentences to annotate, receiving feedback, and retraining the model.
    :param model: The NERModel object to use for prediction and training.
    :type model: NERModel
    :param dataset: The Dataset object to use for managing the data.
    :type dataset: Dataset
    :param config: The configuration for the active learning process.
    :type config: AssistantConf
    """

    def __init__(self, model: NERModel, dataset: Dataset, config: AssistantConf) -> None:
        self._dataset: Dataset = dataset
        self._config: AssistantConf = config
        self._model: NERModel = model
        self._annotated_sentences_count = 0

    def _get_sentence_idx(self) -> int:
        """
        Get the index of the next sentence to annotate.

        :return: The index of the next sentence to annotate.
        :rtype: int
        """
        return 0

    def get_sentence(self, annotated: bool = False) -> Optional[Sentence]:
        """
        Get the next sentence to annotate.

        :param annotated: Whether to return the sentence with annotations or not.
        :type annotated: bool
        :return: The next sentence to annotate.
        :rtype: Optional[Sentence]
        """
        idx = self._get_sentence_idx()
        sentence = self._dataset.get_unlabeled_sentence(idx)
        if sentence is None:
            return None

        if not annotated:
            annotations = [Annotation(words=[Word(word)], label=None) for word in sentence]
            return Sentence(annotations)

        word_indices = self._dataset.map_unlabeled_sentence_to_indices(sentence)
        labels_indices, labels_confidences = self._model.predict_with_confidence(word_indices)

        labels = self._dataset.map_indices_to_labels(labels_indices)
        annotations = self._create_annotations(sentence, labels)
        return Sentence(annotations)

    def _create_annotations(self, sentence: List[str], labels: List[str]) -> List[Annotation]:
        """
        Create annotations from the sentence and labels.

        :param sentence: The sentence to create annotations for.
        :type sentence: List[str]
        :param labels: The labels for the sentence.
        :type labels: List[str]
        :return: The annotations for the sentence.
        :rtype: List[Annotation]
        """
        annotations = []
        for i, label in enumerate(labels):
            if label == DEFAULT_UNLABELED_LABEL:
                annotations.append(Annotation(words=[Word(sentence[i])], label=None))
            elif label.startswith("I-") and i != 0 and labels[i - 1].startswith("B-") and label[2:] == labels[i - 1][2:]:
                annotations[-1].words.append(Word(sentence[i]))
            else:
                label_name = label[2:]
                label_data = next((l for l in self._config.labels if l.label == label_name), None)
                annotations.append(Annotation(words=[Word(sentence[i])], label=label_data))
        return annotations

    def give_feedback(self, sentence: Sentence) -> None:
        """
        Receive feedback on the annotated sentence.

        :param sentence: The annotated sentence.
        :type sentence: Sentence
        """
        converted_sentence = sentence.to_list()
        self._dataset.move_sentence_to_labeled(converted_sentence)
        self._annotated_sentences_count += 1

        if self._annotated_sentences_count >= self._config.sampling_batch_size:
            self._annotated_sentences_count = 0
            self._retrain_model()

    def _retrain_model(self) -> None:
        """
        Retrain the model using the labeled data.

        This method trains the model asynchronously using the labeled data.
        """
        features, target = self._dataset.get_training_data()
        weights = self._dataset.get_weights()

        self._model.train_async(
            features,
            target,
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            class_weights=weights,
        )

    @property
    def labels(self) -> List[LabelData]:
        """
        Get the labels used by the assistant.

        :return: The labels used by the assistant.
        :rtype: List[LabelData]
        """
        return self._config.labels

    @property
    def stats(self) -> Dict[str, int | float]:
        """
        Get the statistics of the assistant.

        :return: The statistics of the assistant.
        :rtype: Dict[str, int | float]
        """
        stats = {
            "label_count": self._dataset.count_labels(),
            "labeled_sentences_count": self._dataset.labeled_sentences_count,
            "unlabeled_sentences_count": self._dataset.unlabeled_sentences_count,
        }

        if self._dataset.labeled_sentences_count > 0:
            features, targets = self._dataset.get_labeled_sentences_converted()
            metrics = self._model.evaluate_metrics(features, targets)
            stats.update(metrics)
        return stats
