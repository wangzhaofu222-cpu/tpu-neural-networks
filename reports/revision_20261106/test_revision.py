import colorsys
import numpy as np
import pytest
from reports.tools.revision_20261106 import hue_assignments,colour_images

def test_balance_determinism_and_role_isolation():
    a,h=hue_assignments(1000,4601);b,j=hue_assignments(1000,4602)
    assert np.array_equal(np.bincount(a),np.full(10,100))
    assert np.array_equal(a,hue_assignments(1000,4601)[0])
    assert not np.array_equal(a,b)
    assert np.max(np.abs(h-(a+.5)/10))<=.015
    assert np.array_equal((h*10).astype(int),a)

def test_hsv_conversion_matches_reference():
    raw=np.random.default_rng(0).integers(0,256,(20,2,2,3),dtype=np.uint8)
    _,h=hue_assignments(20,1);out=colour_images(raw,h)
    for i in range(20):
        for r in range(2):
            for c in range(2):
                v=.25+.65*np.dot(raw[i,r,c].astype(float)/255,[.299,.587,.114])
                expected=np.rint(np.array(colorsys.hsv_to_rgb(h[i],.8,v))*255)
                assert np.max(np.abs(out[i,r,c].astype(float)-expected))<=1

def test_hue_labels_recoverable_without_image_identity():
    a,h=hue_assignments(1000,4603)
    raw=np.random.default_rng(2).integers(0,256,(1000,2,2,3),dtype=np.uint8)
    out=colour_images(raw,h)
    recovered=np.array([int(colorsys.rgb_to_hsv(*(x[0,0]/255))[0]*10) for x in out])
    assert np.array_equal(a,recovered)
    with pytest.raises(ValueError):hue_assignments(11,1)
