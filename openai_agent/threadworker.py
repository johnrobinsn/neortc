# Copyright 2024 John Robinson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import queue

class ThreadWorker:
    def __init__(self,supportOutQ=True):
        self.inQ = queue.Queue()
        self.outQ = queue.Queue() if supportOutQ else None
        self.thread = threading.Thread(target=self.worker)
        self.thread.start()

    def worker(self):
        while True:
            item = self.inQ.get()
            if item is None:
                break
            func, args, kwargs = item
            try:
                result = func(*args, **kwargs)
                if self.outQ and result is not None:
                    self.outQ.put(result)
            except Exception as e:
                if self.outQ:
                    self.outQ.put(e)
            finally:
                self.inQ.task_done()

    def add_task(self, func, *args, **kwargs):
        self.inQ.put((func, args, kwargs))

    def get_result(self):
        return self.outQ.get() if self.outQ else None

    def stop(self):
        self.inQ.put(None)
        self.thread.join()