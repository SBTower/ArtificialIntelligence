import tensorflow as tf
import threading

# Thread body: loop until the coordinator indicates a stop was requested.
# If some condition becomes true, ask the coordinator to stop.
def MyLoop(coord):
  a = 1
  while not coord.should_stop():
    a = a*2
    print "Thread 1: " + str(a)
    if a>1000:
      coord.request_stop()

def MyLoop2(coord):
  a = 1
  while not coord.should_stop():
    a = a*3
    print "Thread 2: " + str(a)
    if a>10000:
      coord.request_stop()

# Main thread: create a coordinator.
coord = tf.train.Coordinator()

# Create 10 threads that run 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord,)), threading.Thread(target=MyLoop2, args=(coord,))]

# Start the threads and wait for all of them to stop.
for t in threads:
  t.start()
coord.join(threads)

