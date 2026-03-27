import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch


def load_module():
    sklearn = types.ModuleType("sklearn")
    feature_extraction = types.ModuleType("sklearn.feature_extraction")
    text = types.ModuleType("sklearn.feature_extraction.text")
    metrics = types.ModuleType("sklearn.metrics")
    pairwise = types.ModuleType("sklearn.metrics.pairwise")
    fuzzywuzzy = types.ModuleType("fuzzywuzzy")
    langdetect = types.ModuleType("langdetect")
    googletrans = types.ModuleType("googletrans")

    class FakeVectorizer:
        def fit(self, questions):
            self.questions = list(questions)
            return self

        def transform(self, items):
            return list(items)

    class FakeSimilarityRow(list):
        pass

    class FakeSimilarityMatrix(list):
        def argmax(self):
            return max(range(len(self[0])), key=self[0].__getitem__)

    def fake_cosine_similarity(user_vec, faq_vectors):
        query = user_vec[0].lower()
        scores = FakeSimilarityRow()
        for question in faq_vectors:
            score = 1.0 if question in query or query in question else 0.0
            scores.append(score)
        return FakeSimilarityMatrix([scores])

    def fake_ratio(a, b):
        a = a.lower()
        b = b.lower()
        return 100 if a in b or b in a else 0

    class FakeTranslator:
        def translate(self, text, src=None, dest=None):
            return types.SimpleNamespace(text=text)

    text.TfidfVectorizer = FakeVectorizer
    pairwise.cosine_similarity = fake_cosine_similarity
    fuzzywuzzy.fuzz = types.SimpleNamespace(ratio=fake_ratio)
    langdetect.detect = lambda text: "en"
    googletrans.Translator = FakeTranslator

    sys.modules.update(
        {
            "sklearn": sklearn,
            "sklearn.feature_extraction": feature_extraction,
            "sklearn.feature_extraction.text": text,
            "sklearn.metrics": metrics,
            "sklearn.metrics.pairwise": pairwise,
            "fuzzywuzzy": fuzzywuzzy,
            "langdetect": langdetect,
            "googletrans": googletrans,
        }
    )

    module_path = Path(__file__).with_name("smart_travel_bot (1).py")
    spec = importlib.util.spec_from_file_location("smart_travel_bot_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = load_module()


def test_known_question_returns_expected_answer():
    answer = module.smart_travel_bot("What tours are available?")
    assert answer == module.faq["what tours are available"]


def test_blank_input_returns_fallback_answer():
    answer = module.smart_travel_bot("   ")
    assert answer == module.FALLBACK_ANSWER


def test_translation_failures_do_not_crash_non_english_queries():
    with patch.object(module, "detect", return_value="es"), patch.object(
        module.translator,
        "translate",
        side_effect=RuntimeError("translation unavailable"),
    ):
        answer = module.smart_travel_bot("¿Qué tours ofrecen?")

    assert isinstance(answer, str)
    assert answer == module.FALLBACK_ANSWER
