import numpy as np
from abc import ABC, abstractmethod
from ..utils import TrackEvalException

class _BaseMetric(ABC):
    @abstractmethod
    def __init__(self):
        self.integer_fields = []
        self.float_fields = []
        self.array_labels = []
        self.integer_array_fields = []
        self.float_array_fields = []
        self.fields = []
        self.summary_fields = []
        self.registered = False

    @abstractmethod
    def eval_sequence(self, data):
        ...

    @classmethod
    def get_name(cls):
        return cls.__name__

    def print_table(self, table_res, tracker, cls):
        metric_name = self.get_name()
        headers = [metric_name] + self.summary_fields
        
        # Collect data for the rows
        rows = []
        for seq, results in sorted(table_res.items()):
            summary_res = self._summary_row(results)
            rows.append([""] + summary_res)
        
        # Calculate max width for each column
        col_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + rows))]
        
        # Create table with headers
        def format_row(row):
            return " | ".join(f"{str(item).ljust(width)}" for item, width in zip(row, col_widths))
        
        separator = "-+-".join("-" * width for width in col_widths)
        header_row = format_row(headers)
        
        print(header_row)
        print(separator)
        for row in rows:
            print(format_row(row))
        print('\n')  # Add two empty lines between tables

    def _summary_row(self, results_):
        vals = []
        for h in self.summary_fields:
            if h in self.float_array_fields:
                vals.append("{0:1.5g}".format(100 * np.mean(results_[h])))
            elif h in self.float_fields:
                vals.append("{0:1.5g}".format(100 * float(results_[h])))
            elif h in self.integer_fields:
                vals.append("{0:d}".format(int(results_[h])))
            else:
                raise NotImplementedError("Summary function not implemented for this field type.")
        return vals

    def _calculate_widths(self, headers, table_res):
        """Calculate the maximum width needed for each column"""
        # Create a list of all rows to be printed
        rows = [headers]
        for results in table_res.values():
            rows.append(self._summary_row(results))
        
        # Calculate the maximum width needed for each column
        widths = [max(len(str(item)) for item in col) for col in zip(*rows)]
        return widths

    def _row_print(self, row, col_widths):
        """Prints results in an evenly spaced row, adapting to the width of each column"""
        to_print = " | ".join(f"{str(item):<{width}}" for item, width in zip(row, col_widths))
        print(to_print)


    def summary_results(self, table_res):
        return dict(zip(self.summary_fields, self._summary_row(list(table_res.values())[0])))

    def detailed_results(self, table_res):
        """Returns detailed final results for a tracker"""
        detailed_fields = self.float_fields + self.integer_fields
        for h in self.float_array_fields + self.integer_array_fields:
            for alpha in [int(100 * x) for x in self.array_labels]:
                detailed_fields.append(h + '___' + str(alpha))
            detailed_fields.append(h + '___AUC')

        detailed_results = {}
        for seq, res in table_res.items():
            detailed_row = self._detailed_row(res)
            if len(detailed_row) != len(detailed_fields):
                raise TrackEvalException(
                    'Field names and data have different sizes (%i and %i)' % (len(detailed_row), len(detailed_fields)))
            detailed_results[seq] = dict(zip(detailed_fields, detailed_row))
        return detailed_results

    def _detailed_row(self, res):
        detailed_row = []
        for h in self.float_fields + self.integer_fields:
            detailed_row.append(res[h])
        for h in self.float_array_fields + self.integer_array_fields:
            for i, alpha in enumerate([int(100 * x) for x in self.array_labels]):
                detailed_row.append(res[h][i])
            detailed_row.append(np.mean(res[h]))
        return detailed_row
