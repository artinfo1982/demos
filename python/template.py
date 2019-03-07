#! /usr/bin/env python3

'''
小工具书写模板
'''

import sys
import getopt


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def func(a, b, c):
    print('%s, %s, %s' % (a, b, c))


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(
                argv[1:], 'ha:b:c:', ['help', 'aaa=', 'bbb=', 'ccc=']
            )
        except getopt.error as msg:
            raise Usage(msg)

        aa = None
        bb = None
        cc = None

        for opt, arg in opts:
            if opt in ('-a', '--aaa'):
                aa = arg
            elif opt in ('-b', '--bbb'):
                bb = arg
            elif opt in ('-c', '--ccc'):
                cc = arg
            elif opt in ('-h', '--help'):
                print(
                    '\nUsage: python %s [OPTIONS]\n\n'
                    '   -a integer, --aaa=XXX.\n'
                    '   -b file, --bbb=XXX.\n'
                    '   -c string, --ccc=XXX.\n'
                    % argv[0]
                )
                return 0

        if not aa:
            raise Usage('aa required.')
        if not bb:
            raise Usage('bb required.')
        if not cc:
            raise Usage('cc required.')

        func(aa, bb, cc)
    except Usage as err:
        print('Error: ' + err.msg, file=sys.stderr)
        print('for help use --help', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
