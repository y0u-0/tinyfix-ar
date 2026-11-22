from tinyfix_ar.corpus import is_clean_saudi_host
from tinyfix_ar.text import normalize, tokenize


def test_normalize_keeps_letters_the_model_fixes():
    assert normalize("أإآؤئءةى") == "أإآؤئءةى"


def test_normalize_strips_diacritics_and_tatweel():
    assert normalize("مَدْرَسَةٌ") == "مدرسة"
    assert normalize("جمـــيل") == "جميل"


def test_normalize_maps_presentation_forms_and_persian_lookalikes():
    assert normalize("ﻻ") == "لا"
    assert normalize("کتابی") == "كتابي"


def test_tokenize_splits_arabic_punctuation_numbers_and_latin():
    assert tokenize("ذهبت، إلى Google في ٢٠٢٤؟") == ["ذهبت", "،", "إلى", "<lat>", "في", "<num>", "؟"]


def test_tokenize_keeps_decimal_numbers_whole():
    assert tokenize("بلغ 3.5 مليون") == ["بلغ", "<num>", "مليون"]


def test_saudi_hosts_keep_saudi_outlets_and_drop_dirty_archives():
    assert is_clean_saudi_host("https://www.spa.gov.sa/w123")
    assert is_clean_saudi_host("https://sabq.org/x")
    assert is_clean_saudi_host("https://uqu.edu.sa/page")
    assert not is_clean_saudi_host("https://www.alyaum.com/articles/1")
    assert not is_clean_saudi_host("https://classic.aawsat.com/x")
    assert not is_clean_saudi_host("https://faculty.ksu.edu.sa/someone")
    assert not is_clean_saudi_host("https://www.aljazeera.net/news")
