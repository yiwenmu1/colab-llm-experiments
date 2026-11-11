"""Execute through `colab exec`, where the Notebook kernel owns Drive mounting."""
from google.colab import drive
drive.flush_and_unmount()
print('Drive flushed and unmounted.')
