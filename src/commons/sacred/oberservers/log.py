#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import logging
from os import system

import visdom
from dateutil import tz
from sacred.observers.base import RunObserver

logger = logging.getLogger(__name__)


def utc_to_local(datetime):
    utc = datetime.replace(tzinfo=tz.tzutc())
    return utc.astimezone(tz.tzlocal())


class LogObserver(RunObserver):

    @staticmethod
    def create(visdom_opts, *args, **kwargs):
        return LogObserver(visdom_opts, *args, **kwargs)

    def __init__(self, visdom_opts, *args, **kwargs):
        self.visdom_opts = visdom_opts
        super(LogObserver, self).__init__(*args, **kwargs)

    def queued_event(self, ex_info, command, host_info, queue_time, config,
                     meta_info, _id):
        logger.info("QUEUED")

    def started_event(self, ex_info, command, host_info, start_time, config,
                      meta_info, _id):
        self.config = config

        # Set the GNU screen pane name to current exp id, should first check if screen is used.
        system("echo '\ek{}\e\\'".format(_id))

        logger.info(config)
        self.config = config

    def heartbeat_event(self, info, captured_out, beat_time, result):
        pass

    def completed_event(self, stop_time, result):
        local_time = utc_to_local(stop_time)
        logger.info('completed_event')

        viz = visdom.Visdom(**self.config['visdom_conf'])
        viz.text('Completed at {}'.format(local_time))

    def interrupted_event(self, interrupt_time, status):
        local_time = utc_to_local(interrupt_time)
        logger.info('interrupted_event')

        viz = visdom.Visdom(**self.config['visdom_conf'])
        viz.text('Interruped at {}'.format(local_time))

    def failed_event(self, fail_time, fail_trace):
        local_time = utc_to_local(fail_time)
        logger.info('failed_event')

        viz = visdom.Visdom(**self.config['visdom_conf'])
        viz.text('Failed at {}\n{}'.format(local_time, fail_trace))

    def log_metrics(self, metrics_by_name, info):
        """Store new measurements to the database.

        Take measurements and store them into
        the metrics collection in the database.
        Additionally, reference the metrics
        in the info["metrics"] dictionary.
        """
        pass
