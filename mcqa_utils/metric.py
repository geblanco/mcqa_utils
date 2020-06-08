
class Metric(object):

    def __init__(self, no_answer):
        self.no_answer = no_answer

    def __call__(self, gold_answers, answers):
        raise NotImplementedError()


class C_at_1(Metric):

    def __call__(self, gold_answers, answers):
        # Eye on types! (int, str...)
        correct = 0
        unanswered = 0
        total = len(gold_answers)
        for gold_ans, ans in zip(gold_answers, answers):
            if gold_ans == ans:
                correct += 1
            elif ans == self.no_answer:
                unanswered += 1
        return (1 / total) * (correct + (correct / total) * unanswered)
