import logging

from tqdm import tqdm

from src.commons.pytorch.evaluation.RecognizeCommands import RecognizeCommands

logger = logging.getLogger(__name__)


class StreamingAccuracyStats(object):
    def __init__(self):
        self.how_many_ground_truth_words = 0
        self.how_many_ground_truth_matched = 0
        self.how_many_false_positives = 0
        self.how_many_correct_words = 0
        self.how_many_wrong_words = 0

    def get_percentages(self):
        return dict(
            any_match_percentage=(self.how_many_ground_truth_matched * 100) / self.how_many_ground_truth_words,
            correct_match_percentage=(self.how_many_correct_words * 100) / self.how_many_ground_truth_words,
            wrong_match_percentage=(self.how_many_wrong_words * 100) / self.how_many_ground_truth_words,
            false_positive_percentage=(self.how_many_false_positives * 100) / self.how_many_ground_truth_words
        )


def get_accuracy_stats(ground_truth, found_words, time_tolerence_ms, up_to_time_ms=-1):
    if up_to_time_ms == -1:
        latest_possible_time = float('inf')
    else:
        latest_possible_time = up_to_time_ms + time_tolerence_ms

    stats = StreamingAccuracyStats()

    for _, truth_time in ground_truth:
        if truth_time > latest_possible_time:
            break
        stats.how_many_ground_truth_words += 1

    has_ground_truth_been_matched = []
    for found_word, found_time in tqdm(found_words, desc='Compute stats'):
        earliest_time = found_time - time_tolerence_ms
        latest_time = found_time + time_tolerence_ms
        has_match_been_found = False
        for truth_word, truth_time in ground_truth:
            if truth_time > latest_time or truth_time > latest_possible_time:
                break
            if truth_time < earliest_time:
                continue
            if truth_word == found_word and truth_time not in has_ground_truth_been_matched:
                stats.how_many_correct_words += 1
            else:
                stats.how_many_wrong_words += 1
            has_ground_truth_been_matched.append(truth_time)
            has_match_been_found = True
            break
        if not has_match_been_found:
            stats.how_many_false_positives += 1
    stats.how_many_ground_truth_matched = len(has_ground_truth_been_matched)
    return stats


def print_accuracy_stats(stats):
    assert stats.how_many_ground_truth_words > 0

    msg = '{any_match_percentage}% matched, ' \
          '{correct_match_percentage}% correctly, ' \
          '{wrong_match_percentage}% wrongly, ' \
          '{false_positive_percentage}% false positives'
    logger.info(msg.format(**stats.get_percentages()))


class StreamingAccuracy(object):
    def __init__(self, labels, clip_duration_ms=1000, clip_stride_ms=30, average_window_ms=500, time_tolerance_ms=750,
                 suppression_ms=1500, minimum_count=3, detection_threshold=0.7):
        self.labels = labels
        self.clip_duration_ms = clip_duration_ms
        self.clip_stride_ms = clip_stride_ms
        self.time_tolerance_ms = time_tolerance_ms
        # self.average_window_ms = average_window_ms
        # self.suppression_ms = suppression_ms
        # self.detection_threshold = detection_threshold

        self.command_recognizer = RecognizeCommands(labels, average_window_ms, detection_threshold, suppression_ms,
                                                    minimum_count)

    def compute_accuracy(self, predictions, ground_truth, up_to_time_ms):
        """

        :param predictions:
        :param ground_truth:
        :return:
        """
        found_words = []
        for pred, time in tqdm(predictions, desc='Recognize commands'):
            found_command, score, is_new_command = self.command_recognizer.process_latest_results(latest_results=pred,
                                                                                                  current_time_ms=time)
            if is_new_command and found_command != '_silence_':
                found_words.append((found_command, time))

        stats = get_accuracy_stats(ground_truth, found_words, self.time_tolerance_ms, up_to_time_ms)
        print_accuracy_stats(stats)
        return stats
