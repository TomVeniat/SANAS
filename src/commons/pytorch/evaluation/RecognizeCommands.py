import numpy as np


class RecognizeCommands(object):
    def __init__(self, labels, average_window_duration_ms=1000, detection_threshold=.2, suppression_ms=500,
                 minimum_count=3, *args, **kwargs):
        """

        :param labels:
        :param average_window_duration_ms:
        :param detection_threshold:
        :param suppression_ms:
        :param minimum_count:
        :param args:
        :param kwargs:
        """
        super(RecognizeCommands, self).__init__(*args, **kwargs)
        self.labels = labels
        self.average_window_duration_ms = average_window_duration_ms
        self.detection_threshold = detection_threshold
        self.suppression_ms = suppression_ms
        self.minimum_count = minimum_count
        self.labels_count = len(labels)
        self.previous_top_label = "_silence_"
        self.previous_top_label_time = float('-inf')

        self.previous_results = []

    def process_latest_results(self, latest_results, current_time_ms):
        assert latest_results.size == self.labels_count
        assert not self.previous_results or current_time_ms > self.previous_results[-1][0]

        self.previous_results.append((current_time_ms, latest_results))

        # Pruning to check
        time_limit = current_time_ms - self.average_window_duration_ms
        while self.previous_results[0][0] < time_limit:
            self.previous_results.pop(0)

        # If there are too few results, assume the result will be unreliable and bail.
        how_many_results = len(self.previous_results)
        earliest_time = self.previous_results[0][0]
        samples_duration = current_time_ms - earliest_time
        if (how_many_results < self.minimum_count) or (samples_duration < (self.average_window_duration_ms / 4)):
            found_command = self.previous_top_label
            score = 0.0
            is_new_command = False
            return found_command, score, is_new_command
            # return Status::OK();

        # Calculate the average score across all the results in the window.
        average_scores = np.zeros(self.labels_count)
        for previous_time, previous_scores in self.previous_results:
            for i in range(self.labels_count):
                average_scores[i] += previous_scores[i] / how_many_results

        # Sort the averaged results in descending score order.
        sorted_average_scores = average_scores.argsort()[::-1]

        current_top_index = sorted_average_scores[0]
        current_top_label = self.labels[current_top_index]
        current_top_score = average_scores[current_top_index]

        if self.previous_top_label == "_silence_" or self.previous_top_label_time == float('-inf'):
            time_since_last_top = float('inf')
        else:
            time_since_last_top = current_time_ms - self.previous_top_label_time

        if current_top_score > self.detection_threshold \
                and current_top_label != self.previous_top_label \
                and time_since_last_top > self.suppression_ms:
            self.previous_top_label = current_top_label
            self.previous_top_label_time = current_time_ms
            is_new_command = True
        else:
            is_new_command = False

        found_command = current_top_label
        score = current_top_score

        return found_command, score, is_new_command


def basic_test():
    rc = RecognizeCommands(["_silence_", "a", "b"])
    results = np.array([1, 0, 0])
    rc.process_latest_results(results, 0)


def find_command_test():
    rc = RecognizeCommands(["_silence_", "a", "b"], 1000, 0.2)
    results = np.array([0, 1, 0])
    has_found_new_command = False
    new_command = None
    for i in range(10):
        current_time_ms = 100 * i
        found_command, score, is_new_command = rc.process_latest_results(results, current_time_ms)
        if is_new_command:
            assert not has_found_new_command
            has_found_new_command = True
            new_command = found_command
    assert has_found_new_command
    assert new_command == "a"

    results = np.array([0, 0, 1])
    has_found_new_command = False
    new_command = None
    for i in range(10):
        current_time_ms = 1000 + (100 * i)
        found_command, score, is_new_command = rc.process_latest_results(results, current_time_ms)
        if is_new_command:
            assert not has_found_new_command
            has_found_new_command = True
            new_command = found_command

    assert has_found_new_command
    assert new_command == "b"

def bad_input_length_test():
    rc = RecognizeCommands(["_silence_", "a", "b"], 1000, 0.2)
    results = np.array([0, 1, 0])
    rc.process_latest_results(results, 100)
    rc.process_latest_results(results, 0)


""" Ugly Testing """
if __name__ == '__main__':
    basic_test()
    find_command_test()
    bad_input_length_test()


