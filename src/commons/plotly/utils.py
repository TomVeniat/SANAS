def get_traceopts_errors(name, errors):
    trace = {name: {'error_y':
                        {'visible': True,
                         'type': 'data',
                         'symmetric': True,
                         'array': errors}}}
    return dict(plotly=trace)
