import datetime
import logging

from src.real.main import *


def app():
    # Possibilities: quic40, quic40_asn, tor
    runExperiment('tor')


if __name__ == '__main__':
    app()
