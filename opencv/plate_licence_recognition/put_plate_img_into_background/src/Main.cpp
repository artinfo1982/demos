#include "PlrIntoSrc.h"

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		printf("ERROR, usage: %s beginIndex numPerImg\n", argv[0]);
		exit(1);
	}
	int index = atoi(argv[1]);
	int numPerImg = atoi(argv[2]);
	int i, j;
	int block_num = SRC_IMAGE_SIDE_LEN / DIV_SIDE_LEN;
	p = (char *)malloc((TOTAL_BLOCK_NUM * 8 * 200 + 300) * sizeof(char));
	buf = (char *)malloc(LOOP_BUF_SIZE * sizeof(char));
	plr_name = (char *)malloc(128 * sizeof(char));

	char *bg_name = (char *)malloc(128 * sizeof(char));
	char *jpg_name = (char *)malloc(128 * sizeof(char));
	char *xml_name = (char *)malloc(128 * sizeof(char));

	for (i = 1; i <= MAX_BG_FILE_NUM; ++i)
	{
		memset(bg_name, 0x0, 128);
		sprintf(bg_name, "%s%d%s", "/home/cd/plr/background/bg_", i, ".jpg");
		for (j = 0; j < numPerImg; ++j)
		{
			memset(jpg_name, 0x0, 128);
			memset(xml_name, 0x0, 128);
			sprintf(jpg_name, "%s%d%s", "/home/cd/plr/imageset/train/", index, ".jpg");
			sprintf(xml_name, "%s%d%s", "/home/cd/plr/xml/", index, ".xml");
			process_single_background_image(bg_name, "/home/cd/plr/licence/", block_num, jpg_name);
			write_label_xml(xml_name, index);
			index++;
		}
	}
	
	return 0;
}
