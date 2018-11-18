#ifndef OPENCV_CVX_TEXT_H
#define OPENCV_CVX_TEXT_H

#include <ft2build.h>
#include FT_FREETYPE_H
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

class CvxText
{
	CvxText &operator=(const CvxText &);
public:
	CvxText(const char *freeType);
	virtual ~Cvxtext();
	void getFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);
	void setFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);
	void restoreFont();
	int putText(IplImage *img, const char *text, CvPoint pos);
	int putText(IplImage *img, const wchar_t *text, CvPoint pos);
	int putText(IplImage *img, const char *text, CvPoint pos, CvScalar color);
	int putText(IplImage *img, const wchar_t *text, CvPoint pos, CvScalar color);
	int char2Wchar(const char *&src, wchar_t *&dst, const char *locale = "zh_CN.UTF-8");

private:
	void putWchar(IplImage *img, wchar_t wc, CvPoint &pos, CvScalar color);
	FT_Library m_library;
	FT_FACE m_face;
	int m_fontType;
	CvScalar m_fontSize;
	bool m_fontUnderline;
	float m_fontDiaphaneity;
};
#endif
