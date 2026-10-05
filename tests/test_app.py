import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


class AppTests(unittest.TestCase):
    def test_demo_does_not_offer_synthetic_or_incomplete_models(self):
        from faceage.inference import available_models
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for name,research,status in [('smoke',False,'complete'),('unfinished',True,'training'),('ready',True,'complete')]:
                folder=root/name; folder.mkdir()
                (folder/'run.json').write_text(json.dumps({'method':'b1','research_model':research,'status':status,
                    'best_validation_mae':5.0,'data_counts':{'train':20}}))
                (folder/'best.pt').write_bytes(b'placeholder for inventory test')
            self.assertEqual([m['name'] for m in available_models(root)],['ready'])

    def test_model_inventory_skips_metadata_that_would_crash_the_demo(self):
        from faceage.inference import available_models
        with TemporaryDirectory() as directory:
            root=Path(directory); folder=root/'broken'; folder.mkdir()
            (folder/'run.json').write_text(json.dumps({'method':'b1','research_model':True,'status':'complete'}))
            (folder/'best.pt').write_bytes(b'placeholder')
            self.assertEqual(available_models(root),[])

    def test_uploaded_image_is_decoded_without_writing_a_file(self):
        from faceage.inference import decode_image
        buffer=io.BytesIO(); Image.new('RGB',(50,60),'red').save(buffer,format='PNG')
        self.assertEqual(decode_image(buffer.getvalue()).size,(50,60))
        for content in [b'', b'x'*(10*1024*1024+1)]:
            with self.assertRaises(ValueError): decode_image(content)

    def test_streamlit_empty_state_renders_without_a_fake_prediction(self):
        path=Path(__file__).resolve().parents[1]/'app.py'
        self.assertTrue(path.is_file(),'Demo Streamlit belum dibuat')
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(path)).run(timeout=30)
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('belum tersedia' in item.value.lower() for item in app.info))
        self.assertEqual(len(app.metric),0)
