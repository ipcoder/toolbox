import ipywidgets as widgets
import logging
from IPython.display import display


class OutputWidgetHandler(logging.Handler):
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, logs=None, *args, width='95%', height='95%', log_lines=10,
                 formatter='[%(levelname)s] %(message)s', propagate=False, **kwargs):

        """
        :param log: logger or its name to attache the handler to
        :param width: width of the handler window
        :param maximal: number of visible lines
        :param formatter: formatter
        :param propagate: to control propagation of the log - avoids double log messages in some cases
        :params *args, **kwargs: Handler arguments
        """

        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        self.out = widgets.Output(layout=dict(width=width, height=height, border='1px solid green'))
        self.log_lines = log_lines-1

        if isinstance(formatter, str):
            formatter = logging.Formatter(formatter)
        self.setFormatter(formatter)

        if logs is None:
            logs = [logging.getLogger()]
        elif isinstance(logs, str):
            logs = [logging.getLogger(logs)]
        elif isinstance(logs, logging.Logger):
            logs = [logs]
        else:
            logs = [logging.getLogger(log) if isinstance(log, str) else log for log in logs]

        for log in logs:
            log.handlers = []
            log.propagate = propagate
            if 'level' in kwargs:
                log.setLevel(kwargs['level'])
            log.addHandler(self)

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record+'\n'
        }
        self.out.outputs = self.out.outputs[-self.log_lines:] + (new_output, )

    def show_logs(self):
        """ Show the logs """
        display(self.out)

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()

# logger = logging.getLogger(__name__)
# handler = OutputWidgetHandler()
# handler.setFormatter(logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)
