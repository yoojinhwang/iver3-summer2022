from hydrophone import Hydrophone
import time
import sys

if __name__ == '__main__':
    # Read in command line arguments: port, serial_no, savepath to a file to dump data
    _, port, serial_no, *rest = sys.argv
    if len(rest) != 0:
        savepath = rest[0]
    else:
        savepath = None
    
    # Create the hydrophone object
    h1 = Hydrophone(port, int(serial_no), savepath=savepath)

    # Start hydrophone
    h1.start()
    while h1.is_starting():
        h1.run()
        time.sleep(0.01)

    if not h1.is_closed():
        # Listen for pings
        while True:
            try:
                h1.run()
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
        
        # Stop the hydrophone
        h1.stop()
        while h1.is_stopping():
            h1.run()
            time.sleep(0.01)
        h1.close()