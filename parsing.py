class Parsing(object):

    @staticmethod
    def parse_wtag_file_to_lists(file_full_name: str) -> (list, list):
        with open(file_full_name, 'r') as f:
            lines = f.readlines()

        sentences = []
        tags = []
        # For each line, creates a list of word_POS
        raw_sentences = [line.split(' ') for line in lines]
        for sentence in raw_sentences:
            # For each word, creates a list of [word, POS]
            sliced_sentence = [word_tag.split('_') for word_tag in sentence]
            sentences.append([word_tag[0] for word_tag in sliced_sentence])
            tags.append([word_tag[1].rstrip() for word_tag in sliced_sentence])
        return sentences, tags

    @staticmethod
    def parse_wtag_file_to_tags(file_full_name: str) -> list:
        with open(file_full_name, 'r') as f:
            lines = f.readlines()

        tags = set()
        # For each line, creates a list of word_POS
        raw_sentences = [line.split(' ') for line in lines]
        for sentence in raw_sentences:
            # For each word, creates a list of [word, POS]
            sliced_sentence = [word_tag.split('_') for word_tag in sentence]
            tags |= {word_tag[1].rstrip() for word_tag in sliced_sentence}
        return sorted(tags)

    def parse_wtag_file_to_words_file(self, wtag_file_full_name: str):
        sentences, _ = self.parse_wtag_file_to_lists(wtag_file_full_name)
        words_file_full_name = wtag_file_full_name.replace("wtag", "words")
        with open(words_file_full_name, 'w+') as f:
            len_file = len(sentences)
            for sentence_idx, sentence in enumerate(sentences):
                len_sentence = len(sentence)
                for word_idx, word in enumerate(sentence):
                    if word_idx < len_sentence - 1:
                        print(word, end=' ', file=f)
                    else:
                        print(word, end='', file=f)
                if sentence_idx < len_file - 1:
                    print(file=f)

    @staticmethod
    def parse_words_file_to_list(file_full_name: str) -> list:
        with open(file_full_name) as f:
            lines = f.readlines()

        lines = [line.split(' ') for line in lines]
        sentences = []
        for sentence in lines:
            sentences.append([word.rstrip() for word in sentence])
        return sentences

    @staticmethod
    def parse_lists_to_wtag_file(sentences: list, tags: list, file_full_name: str):
        with open(file_full_name, 'w+') as f:
            num_of_sentences = len(sentences)
            for idx_sentence, (sentence, sentence_tags) in enumerate(zip(sentences, tags)):
                sentence_length = len(sentence)
                for idx_word, (word, tag) in enumerate(zip(sentence, sentence_tags)):
                    f.write(word + "_" + tag)
                    # Puts space after each word_POS but the last one
                    if idx_word != sentence_length - 1:
                        f.write(" ")
                # Puts new line after each sentence but the last one
                if idx_sentence != num_of_sentences - 1:
                    f.write("\n")
