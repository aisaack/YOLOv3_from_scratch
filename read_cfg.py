def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    holder = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()

            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)

    return blocks

if __name__ == '__main__':
    blocks = parse_cfg('./yolov3.cfg')
    for block in blocks:
        print(block)
