import torch


def boxes_iou(box1, box2):
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
    Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)
    union_width = Mx - mx
    my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
    My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)
    union_height = My - my
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    intersection_area = intersection_width * intersection_height
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area/union_area
    return iou


def nms(boxes, iou_thresh):
    if len(boxes) == 0:
        return boxes
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]
    _, sortIds = torch.sort(det_confs, descending=True)
    best_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            best_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0
    return best_boxes


def detect_objects(model, img, iou_thresh, nms_thresh):
    model.eval()
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)
    img = img.unsqueeze(0)
    list_boxes = model(img, nms_thresh)
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    boxes = nms(boxes, iou_thresh)
    return boxes


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def print_objects(boxes, class_names):
    print('Objects Found and Confidence Level:\n')
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))


def num_of_person(boxes, class_names):
    num = 0
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_id = box[6]
            if class_names[cls_id] == "person":
                num += 1
    return num
