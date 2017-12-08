class Parsing(object):

    @staticmethod
    # TODO: save also \n in the last character. Remember that!
    def parse_wtag_file_to_lists(file_full_name: str) -> (list, list):
        with open(file_full_name) as f:
            lines = f.readlines()

        sentences = []
        tags = []
        # For each line, creates a list of word_POS
        raw_sentences = [line.split(' ') for line in lines]
        for sentence in raw_sentences:
            # For each word, creates a list of [word, POS]
            sliced_sentence = [word_tag.split('_') for word_tag in sentence]
            sentences.append([word_tag[0] for word_tag in sliced_sentence])
            tags.append([word_tag[1] for word_tag in sliced_sentence])
        return sentences, tags

    @staticmethod
    def parse_words_file_to_list(file_full_name: str) -> list:
        with open(file_full_name) as f:
            lines = f.readlines()

        return [line.split(' ') for line in lines]

    @staticmethod
    def parse_lists_to_wtag_file(sentences: list, tags: list, file_full_name: str):
        with open(file_full_name, 'w') as f:
            for sentence, sentence_tags in zip(sentences, tags):
                sentence_length = len(sentence)
                for idx, (word, tag) in enumerate(zip(sentence, sentence_tags)):
                    f.write(word + "_" + tag)
                    # Puts space after each word_POS but the last one
                    if idx != sentence_length - 1:
                        f.write(" ")
