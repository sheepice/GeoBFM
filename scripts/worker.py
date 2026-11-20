"""Internal torchrun target; public commands go through the GPU-owning launcher."""
import _bootstrap
import runpy
import sys

if __name__ == "__main__":
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name="__main__")
