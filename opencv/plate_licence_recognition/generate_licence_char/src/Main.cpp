#include "PlrChrGen.h"

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
	init_zhs_eng_font("/home/cd/plr/font/platech.ttf", "/home/cd/plr/font/platechar.ttf");
	init_image_rect_memory();
	init_image_rect_color();

	int block_num = SRC_IMAGE_SIDE_LEN / DIV_SIDE_LEN;
	total_block_num = block_num * block_num;
	p = (char *)malloc((total_block_num * 200 + 300) * sizeof(char));
	buf = (char *)malloc(512 * sizeof(char));
	class_array = (int *)malloc(total_block_num * sizeof(int));
	x_array = (int *)malloc(total_block_num * sizeof(int));
	y_array = (int *)malloc(total_block_num * sizeof(int));

	char *bg_name = (char *)malloc(128 * sizeof(char));
	char *jpg_name = (char *)malloc(128 * sizeof(char));
	char *xml_name = (char *)malloc(128 * sizeof(char));

	for (i = 1; i <= 50; ++i)
	{
		memset(bg_name, 0x0, 128);
		sprintf(bg_name, "%s%d%s", "/home/cd/plr/background/bg_", i, ".jpg");
		for (j = 0; j < numPerImg; ++j)
		{
			memset(jpg_name, 0x0, 128);
			memset(xml_name, 0x0, 128);
			sprintf(jpg_name, "%s%d%s", "/home/cd/plr/imageset/train/", index, ".jpg");
			sprintf(xml_name, "%s%d%s", "/home/cd/plr/xml/", index, ".xml");
			process_single_background_image(bg_name, block_num, jpg_name);
			write_label_xml(xml_name, index);
			index++;
		}
	}
	release_all();
	free(p);
	p = NULL;
	return 0;
}
