import os


class PathConfig:
    def __init__(self, base_dir="./sentence_20"):
        # 基础目录
        self.base_dir = base_dir

        # 句子目录
        self.sentence_dir = os.path.join(base_dir, "sentence_candidates")
        self.sentence_ppl_dir = os.path.join(base_dir, "sentence_ppl")
        self.sentence_simcse_dir = os.path.join(base_dir, "sentence_simcse")
        self.sentence_probability_dir = os.path.join(base_dir, "sentence_probability")

        # 具体文件路径
        self.question_candidates = os.path.join(self.sentence_dir, "question_candidates.json")
        self.sentence_candidates = os.path.join(self.sentence_dir, "sentence_candidate.json")

        self.question_ppl = os.path.join(self.sentence_ppl_dir, "question_candidates_ppl_10000.json")
        self.sentence_ppl = os.path.join(self.sentence_ppl_dir, "sentence_candidates_ppl_10000.json")

        self.question_ppl_filter = os.path.join(self.sentence_ppl_dir, "question_candidates_ppl_filter_5000.json")
        self.sentence_ppl_filter = os.path.join(self.sentence_ppl_dir, "sentence_candidates_ppl_filter_5000.json")

        self.question_simcse = os.path.join(self.sentence_simcse_dir, "question_candidates_simcse_5000.json")
        self.sentence_simcse = os.path.join(self.sentence_simcse_dir, "sentence_candidates_simcse_5000.json")

        self.question_probability = os.path.join(self.sentence_probability_dir, "question_candidates_probability.json")
        self.sentence_probability = os.path.join(self.sentence_probability_dir, "sentence_candidates_probability.json")


class trainPathConfig:
    def __init__(self, base_dir="./train_sentence"):
        # 基础目录
        self.base_dir = base_dir

        # 句子目录
        self.sentence_dir = os.path.join(base_dir, "sentence_candidates")
        self.sentence_ppl_dir = os.path.join(base_dir, "sentence_ppl")
        self.sentence_simcse_dir = os.path.join(base_dir, "sentence_simcse")
        self.setence_probability_dir = os.path.join(base_dir, "sentence_probability")

        # 具体文件路径
        self.question_candidates = os.path.join(self.sentence_dir, "question_candidates_ppl.json")
        self.sentence_candidates = os.path.join(self.sentence_dir, "sentence_candidates_ppl.json")

        self.question_ppl = os.path.join(self.sentence_ppl_dir, "question_candidates_ppl.json")
        self.sentence_ppl = os.path.join(self.sentence_ppl_dir, "sentence_candidates_ppl.json")

        self.question_ppl_filter = os.path.join(self.sentence_ppl_dir, "question_candidates_ppl_filter.json")
        self.sentence_ppl_filter = os.path.join(self.sentence_ppl_dir, "sentence_candidates_ppl_filter.json")

        self.question_simcse = os.path.join(self.sentence_simcse_dir, "question_candidates_simcse_20.json")
        self.sentence_simcse = os.path.join(self.sentence_simcse_dir, "sentence_candidates_simcse_20.json")

        self.question_probability = os.path.join(self.sentence_simcse_dir, "question_probability_probability.json")
        self.sentence_probability = os.path.join(self.sentence_simcse_dir, "sentence_probability_probability.json")
