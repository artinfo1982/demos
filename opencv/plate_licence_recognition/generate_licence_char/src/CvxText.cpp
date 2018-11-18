#include <wchar.h>
#include <assert.h>
#include <locale.h>
#include <ctype.h>
#include "CvxText.h"

CvxText::CvxText(const char *freeType)
{
	assert(freeType != NULL);
	if (FT_Init_FreeType(&m_library))
		throw;
	if (FT_New_Face(m_library, freeType, 0, &m_face))
		throw;
	restoreFont();
	setFont(LC_CTYPE, "");
}

CvxText::~CvxText()
{
	FT_Done_Face(m_face);
	FT_Done_FreeType(m_library);
}

void CvxText::getFont(int *type, CvScalar *size, bool *underline, float *diaphaneity)
{
	if (type)
		*type = m_fontType;
	if (size)
		*size = m_fontSize;
	if (underline)
		*underline = m_fontUnderline;
	if (diaphaneity)
		*diaphaneity = m_fontDiaphaneity;
}

void CvxText::setFont(int *type, CvScalar *size, bool *underline, float *diaphaneity)
{
	if (type)
	{
		if (type >= 0)
			m_fontType = *type;
	}
	if (size)
	{
		m_fontSize.val[0] = fabs(size->val[0]);
		m_fontSize.val[1] = fabs(size->val[1]);
		m_fontSize.val[2] = fabs(size->val[2]);
		m_fontSize.val[3] = fabs(size->val[3]);
	}
	if (underline)
		m_fontUnderline = *underline;
	if (diaphaneity)
		m_fontDiaphaneity = *diaphaneity;
	FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}

void CvxText::restoreFont()
{
	m_fontType = 0;
	m_fontSize.val[0] = 20;
	m_fontSize.val[1] = 0.5;
	m_fontSize.val[2] = 0.1;
	m_fontSize.val[3] = 0;
	m_fontUnderline = false;
	m_fontDiaphaneity = 1.0;
	FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}

int CvxText::putText(IplImage *img, const char *text, CvPoint pos)
{
	return putText(img, text, pos, CV_RGB(255, 255, 255));
}

int CvxText::putText(IplImage *img, const wchar_t *text, CvPoint pos)
{
	return putText(img, text, pos, CV_RGB(255, 255, 255));
}

int CvxText::putText(IplImage *img, const char *text, CvPoint pos, CvScalar color)
{
	if (NULL == img)
		return -1;
	if (NULL == text)
		return -1;
	wchar_t *w_str;
	int count = char2Wchar(text, w_str);
	int i = 0;
	for (; i < count; ++i)
	{
		wchar_t wc = w_str[i];
		if (wc < 128)
			FT_Set_Pixel_Sizes(m_face, (FT_UInt)(m_fontSize.val[0] * 1.15), 0);
		else
			FT_Set_Pixel_Sizes(m_face, (FT_UInt)(m_fontSize.val[0]), 0);
		putWchar(img, wc, pos, color);
	}
	delete (w_str);
	return i;
}

int CvxText::putText(IplImage *img, const wchar_t *text, CvPoint pos, CvScalar color)
{
	if (NULL == img)
		return -1;
	if (NULL == text)
		return -1;
	int i;
	for (i = 0; text[i] != '\0'; ++i)
		putWchar(img, text[i], pos, color);
	return i;
}

int CvxText::char2Wchar(const char *& src, wchar_t *& dst, const char *locale)
{
	if (NULL == src)
	{
		dst = NULL;
		return 0;
	}
	setlocale(LC_CTYPE, locale);
	int w_size = mbstowcs(NULL, src, 0) + 1;
	if (w_size == 0)
	{
		dst = NULL;
		return -1;
	}
	dst = new wchar_t[w_size];
	if (NULL == dst)
		return -1;
	int ret = mbstowcs(dst, src, strlen(src) + 1);
	if (ret <= 0)
		return -1;
	return ret;
}

void CvxText::putWchar(IplImage *img, wchar_t wc, CvPoint &pos, CvScalar color)
{
	FT_UInt glyph_index = FT_Get_Char_Index(m_face, wc);
	FT_Load_Glyph(m_face, glyph_index, FT_LOAD_DEFAULT);
	FT_Render_Glyph(m_face->glyph, FT_RENDER_MODE_MONO);
	FT_GlyphSlot slot = m_face->glyph;

	int rows = slot->bitmap.rows;
	int cols = slot->bitmap.width;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			int off = ((img->origin == 0) ? i : (rows -1 - i)) * slot->bitmap.pitch + j / 8;
			if (slot->bitmap.buffer[off] & (0xC0 >> (j % 8)))
			{
				int r = (img->origin == 0) ? pos.y - (rows - 1 - i) : pos.y + i;
				int c = pos.x + j;
				if (r >= 0 && r < img->height && c >= 0 && c <= img->width)
				{
					CvScalar scalar = cvGet2D(img, r, c);
					float p = m_fontDiaphaneity;
					for (int k = 0; k < 4; ++k)
						scalar.val[k] = scalar.val[k] * (1 - p) + color.val[k] * p;
					cvSet2D(img, r, c, scalar);
				}
			}
		}
	}

	double space = m_fontSize.val[0] * m_fontSize.val[1];
	double sep = m_fontSize.val[0] * m_fontSize.val[2];
	pos.x += (int)((cols ? cols : space) + sep);
}
