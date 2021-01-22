# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/9/11 11:17 上午   
# 描述：选择运行环境，需要维护env_list\func_dic\config.ini（存储环境默认参数）

from env.snakes import SnakeEatBeans
import configparser
import os


def make(env_type, conf=None):
    config = configparser.ConfigParser()
    path = os.path.join(os.path.dirname(__file__), 'config.ini')
    # print(path)
    config.read(path, encoding="utf-8")
    env_list = config.sections()
    conf_dic = {}
    for env_name in env_list:
        conf_dic[env_name] = config[env_name]
    if env_type not in env_list:
        raise Exception("可选环境列表：%s,传入环境为%s" % (str(env_list), env_type))
    if conf:
        conf_dic[env_type] = conf

    name = env_type.split('_')[0]

    if name == "snakes":
        env = SnakeEatBeans(conf_dic[env_type])
    return env


if __name__ == "__main__":
    make("sokoban_1p")

