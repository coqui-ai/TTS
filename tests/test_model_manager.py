# #!/usr/bin/env python3`
# import os
# import shutil
# import glob
# from tests import get_tests_output_path
# from TTS.utils.manage import ModelManager


# def test_if_all_models_available():
#     """Check if all the models are downloadable."""
#     print(" > Checking the availability of all the models under the ModelManager.")
#     manager = ModelManager(output_prefix=get_tests_output_path())
#     model_names = manager.list_models()
#     for model_name in model_names:
#         manager.download_model(model_name)
#         print(f" | > OK: {model_name}")

#     folders = glob.glob(os.path.join(manager.output_prefix, '*'))
#     assert len(folders) == len(model_names)
#     shutil.rmtree(manager.output_prefix)
