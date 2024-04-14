from data.data_processing import (
    create_unlabeled_dataset,
)

if __name__ == "__main__":
    create_unlabeled_dataset("dataset/ner_dataset.csv")
    # create_word_to_idx('dataset/unlabeled.csv',
    #                    'dataset/word2idx.json')
    # create_tag_to_idx(['per', 'geo'], 'dataset/tag2ixd.json')
