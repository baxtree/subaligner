#!/usr/bin/env python
# -*- encoding=utf8 -*-
#
# stl2srt A program to convert EBU STL subtitle files in the more common SRT format
#
# Copyright 2014 Yann Coupin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
import codecs
import logging
import unicodedata
from xml.etree import ElementTree
import re


class SRT:
    '''A class that behaves like a file object and writes an SRT file'''
    def __init__(self, pathOrFile):
        self.file = open(pathOrFile, 'wb')
        self.counter = 1
        self.file.write(codecs.BOM_UTF8)

    def _formatTime(self, timestamp):
        return "%02u:%02u:%02u,%03u" % (
            timestamp / 3600,
            (timestamp / 60) % 60,
            timestamp % 60,
            (timestamp * 1000) % 1000
        )

    def write(self, start, end, text, encoding):
        text = "\n".join([x for x in text.split("\n") if bool(x)])
        self.file.write(("%0u\n%s --> %s\n%s\n\n" % (self.counter, self._formatTime(start), self._formatTime(end), text)).encode(encoding, errors="replace"))
        self.counter += 1

class iso6937(codecs.Codec):
    '''A class to implement the somewhat exotic iso-6937 encoding which STL files often use'''

    identical = set(range(0x20, 0x7f))
    identical |= set((0xa, 0xa0, 0xa1, 0xa2, 0xa3, 0xa5, 0xa7, 0xab, 0xb0, 0xb1, 0xb2, 0xb3, 0xb5, 0xb6, 0xb7, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf))
    direct_mapping = {
        0x8a: 0x000a, # line break

        0xa8: 0x00a4, # ¤
        0xa9: 0x2018, # ‘
        0xaa: 0x201C, # “
        0xab: 0x00AB, # «
        0xac: 0x2190, # ←
        0xad: 0x2191, # ↑
        0xae: 0x2192, # →
        0xaf: 0x2193, # ↓

        0xb4: 0x00D7, # ×
        0xb8: 0x00F7, # ÷
        0xb9: 0x2019, # ’
        0xba: 0x201D, # ”
        0xbc: 0x00BC, # ¼
        0xbd: 0x00BD, # ½
        0xbe: 0x00BE, # ¾
        0xbf: 0x00BF, # ¿

        0xd0: 0x2015, # ―
        0xd1: 0x00B9, # ¹
        0xd2: 0x00AE, # ®
        0xd3: 0x00A9, # ©
        0xd4: 0x2122, # ™
        0xd5: 0x266A, # ♪
        0xd6: 0x00AC, # ¬
        0xd7: 0x00A6, # ¦
        0xdc: 0x215B, # ⅛
        0xdd: 0x215C, # ⅜
        0xde: 0x215D, # ⅝
        0xdf: 0x215E, # ⅞

        0xe0: 0x2126, # Ohm Ω
        0xe1: 0x00C6, # Æ
        0xe2: 0x0110, # Đ
        0xe3: 0x00AA, # ª
        0xe4: 0x0126, # Ħ
        0xe6: 0x0132, # Ĳ
        0xe7: 0x013F, # Ŀ
        0xe8: 0x0141, # Ł
        0xe9: 0x00D8, # Ø
        0xea: 0x0152, # Œ
        0xeb: 0x00BA, # º
        0xec: 0x00DE, # Þ
        0xed: 0x0166, # Ŧ
        0xee: 0x014A, # Ŋ
        0xef: 0x0149, # ŉ

        0xf0: 0x0138, # ĸ
        0xf1: 0x00E6, # æ
        0xf2: 0x0111, # đ
        0xf3: 0x00F0, # ð
        0xf4: 0x0127, # ħ
        0xf5: 0x0131, # ı
        0xf6: 0x0133, # ĳ
        0xf7: 0x0140, # ŀ
        0xf8: 0x0142, # ł
        0xf9: 0x00F8, # ø
        0xfa: 0x0153, # œ
        0xfb: 0x00DF, # ß
        0xfc: 0x00FE, # þ
        0xfd: 0x0167, # ŧ
        0xfe: 0x014B, # ŋ
        0xff: 0x00AD, # Soft hyphen
    }
    diacritic = {
        0xc1: 0x0300, # grave accent
        0xc2: 0x0301, # acute accent
        0xc3: 0x0302, # circumflex
        0xc4: 0x0303, # tilde
        0xc5: 0x0304, # macron
        0xc6: 0x0306, # breve
        0xc7: 0x0307, # dot
        0xc8: 0x0308, # umlaut
        0xca: 0x030A, # ring
        0xcb: 0x0327, # cedilla
        0xcd: 0x030B, # double acute accent
        0xce: 0x0328, # ogonek
        0xcf: 0x030C, # caron
    }


    def decode(self, input):
        output = []
        state = None
        count = 0
        for char in input:
            char = ord(char)
            # End of a subtitle text
            count += 1
            if not state and char in self.identical:
                output.append(char)
            elif not state and char in self.direct_mapping:
                output.append(self.direct_mapping[char])
            elif not state and char in self.diacritic:
                state = self.diacritic[char]
            elif state:
                combined = unicodedata.normalize('NFC', chr(char) + chr(state))
                if combined and len(combined) == 1:
                    output.append(ord(combined))
                state = None
        return (''.join(map(chr, output)), len(input))

    def search(self, name):
        if name in ('iso6937', 'iso_6937-2'):
            return codecs.CodecInfo(self.encode, self.decode, name='iso_6937-2')

    def encode(self, input):
        pass

codecs.register(iso6937().search)

class RichText:

    def __init__(self, use_html_tags):
        self.tag_stack = []
        self.opened_tags = set()
        self.output = []
        self.add_html_tags = use_html_tags

    def write(self, string):
        self.output.append(string)

    def openTag(self, tag_name, tag_html=None):
        if not tag_html:
            tag_html = '<%s>' % tag_name
        if tag_name not in self.opened_tags:
            self.tag_stack.append((tag_name, tag_html))
            self.opened_tags.add(tag_name)
            if not self.add_html_tags:
                self.output.append(' ')
            else:
                self.output.append(tag_html)

    def closeTag(self, tag):
        if not self.add_html_tags:
            return
        tag_html = '</%s>' % tag
        if tag in self.opened_tags:
            reopen_stack = []
            while self.tag_stack:
                tag_to_close = self.tag_stack.pop()
                if tag_to_close[0] == tag:
                    self.output.append(tag_html)
                    self.opened_tags.remove(tag)
                    break
                else:
                    reopen_stack += tag_to_close
            for tag_to_reopen in reopen_stack:
                self.output.append(tag_to_reopen[1])
                self.tag_stack.append(tag_to_reopen)

    def __str__(self):
        if not self.add_html_tags:
            return ''.join(self.output)

        closing_tags = []
        # Close all the tags still open
        for tag in self.tag_stack[::-1]:
            closing_tags.append('</%s>' % tag[0])
        return ''.join(self.output + closing_tags)

class STL:
    '''A class that behaves like a file object and reads an STL file'''

    GSIfields = 'CPN DFC DSC CCT LC OPT OET TPT TET TN TCD SLR CD RD RN TNB TNS TNG MNC MNR TCS TCP TCF TND DSN CO PUB EN ECD UDA'.split(' ')
    TTIfields = 'SGN SN EBN CS TCIh TCIm TCIs TCIf TCOh TCOm TCOs TCOf VP JC CF TF'.split(' ')

    def __init__(self, pathOrFile, richFormatting=False):
        self.file = open(pathOrFile, 'rb')

        self.richFormatting = richFormatting

        self._readGSI()

    def __bcdTimestampDecode(self, timestamp):
        # Special case for people that can't bother to read a spec
        if timestamp == '________':
            return 0.0

        # BCD coded time with limited significant bits as per EBU Tech. 3097-E
        safe_bytes = [x[0]&x[1] for x in zip((0x2, 0xf, 0x7, 0xf, 0x7, 0xf, 0x3, 0xf), struct.unpack('8B', timestamp))]
        return sum([x[0]*x[1] for x in zip((36000, 3600, 600, 60, 10, 1, 10.0 / self.fps, 1.0 / self.fps), safe_bytes)])

    def _readGSI(self):
        self.GSI = dict(list(zip(
            self.GSIfields,
            struct.unpack('3s8sc2s2s32s32s32s32s32s32s16s6s6s2s5s5s3s2s2s1s8s8s1s1s3s32s32s32s75x576s', self.file.read(1024))
        )))
        GSI = self.GSI
        # GSI ={k: v.decode("utf-8")  for k, v in self.GSI.items()}
        logging.debug(GSI)
        #self.gsiCodePage = 'cp%s' % GSI['CPN']
        if GSI['DFC'] == b'STL24.01':
            self.fps = 24
        elif GSI['DFC'] == b'STL25.01':
            self.fps = 25
        elif GSI['DFC'] == b'STL30.01':
            self.fps = 30
        else:
            raise Exception('Invalid DFC')
        self.codePage = {
            '00': 'iso_6937-2',
            '01': 'iso-8859-5',
            '02': 'iso-8859-6',
            '03': 'iso-8859-7',
            '04': 'iso-8859-8',
        }[GSI['CCT'].decode('utf-8')]
        self.numberOfTTI = int(GSI['TNB'])
        if GSI['TCS'] == '1':
            # BCD coded time with limited significant bits

            self.startTime = self.__bcdTimestampDecode(GSI['TCP'])
        else:
            self.startTime = 0.0
        logging.debug(self.__dict__)

    def __timecodeDecode(self, h, m, s, f):
        return 3600 * h + 60 * m + s + float(f) / self.fps

    def __parseFormatting(self, text, addHtmlTags):
        colorCodes = [
            '#000000', # black
            '#ff0000', # red
            '#00ff00', # green
            '#ffff00', # yellow
            '#0000ff', # blue
            '#ff00ff', # magenta
            '#00ffff', # cyan
            '#ffffff', # white
        ]
        currentColor = 7 # White is the default color
        output = RichText(addHtmlTags)

        first_line = True

        for char in "".join(map(chr, text)):
            ochar = ord(char)
            if ochar == 0x80:
                output.openTag('i')
            elif ochar == 0x81:
                output.closeTag('i')
            elif ochar == 0x82:
                output.openTag('u')
            elif ochar == 0x83:
                output.closeTag('u')
            elif ochar == 0xe:
                output.openTag('b')
            elif ochar == 0xc:
                output.closeTag('b')
            elif ochar in (0,1,2,3,4,5,6,7,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17):
                color = ochar % 0x10
                if color != currentColor:
                    currentColor = color
                    output.closeTag('font')
                    output.openTag('font', '<font color="%s">' % colorCodes[color])
            elif ochar == 0x8a and first_line:
                output.write("\n")
                first_line = False
            elif ochar == 0x8f:
                break
            elif (ochar & 0x7F) >= 0x20:
                output.write(char)

        return str(output)

    def _readTTI(self):
        while (True):
            tci = None
            tco = None
            txt = []

            while (True):
                data = self.file.read(128)
                if not data:
                    raise StopIteration()
                TTI = dict(list(zip(
                    self.TTIfields,
                    struct.unpack('<BHBBBBBBBBBBBBB112s', data)
                )))
                logging.debug(TTI)
                # if comment skip
                if TTI['CF']:
                    continue
                # discard blocks with user data
                if TTI['EBN'] == 254:
                    continue
                if not tci:
                    tci = self.__timecodeDecode(TTI['TCIh'], TTI['TCIm'], TTI['TCIs'], TTI['TCIf']) - self.startTime
                    tco = self.__timecodeDecode(TTI['TCOh'], TTI['TCOm'], TTI['TCOs'], TTI['TCOf']) - self.startTime
                text = TTI['TF']
                text = self.__parseFormatting(text, self.richFormatting)
                text = text.strip()
                txt += text
                if TTI['EBN'] == 255:
                    # skip empty subtitles and those before the start of the show
                    if txt and tci >= 0:
                        return (tci, tco, ''.join(txt))
                    break

    def __iter__(self):
        return self

    def __next__(self):
        return self._readTTI()

class TT:
    '''A class that behaves like a file object and reads a TT subtitle file'''

    __named_colors = {
        'black': '#000000',
        'silver': '#c0c0c0',
        'gray': '#808080',
        'white': '#ffffff',
        'maroon': '#800000',
        'red': '#ff0000',
        'purple': '#800080',
        'fuchsia': '#ff00ff',
        'magenta': '#ff00ff',
        'green': '#008000',
        'lime': '#00ff00',
        'olive': '#808000',
        'yellow': '#ffff00',
        'navy': '#000080',
        'blue': '#0000ff',
        'teal': '#008080',
        'aqua': '#00ffff',
        'cyan': '#00ffff',
    }

    def __init__(self, source, richFormatting):
        self.source = source

        self.richFormatting = richFormatting
        self.frameRate = 30.0
        self.frameRateMultiplier = 1.0
        self.subFrameRate = 1.0
        self.tickRate = None

        self._parseXML()

    def __parse_style(self, element):
        style = {}
        for k, v in list(element.items()):
            if k == '{tts}color':
                if re.match(r'#\d{6}\d{2}?', v):
                    style['color'] = v[0:7]
                else:
                    rgb_color = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', v.strip())
                    if rgb_color:
                        style['color'] = '#%02x%02x%02x' % (int(rgb_color.group(1)), int(rgb_color.group(2)), int(rgb_color.group(3)))
                    else:
                        c = self.__named_colors.get(v)
                        if c:
                            style['color'] = c
            elif k == '{tts}fontStyle':
                style['italic'] = v == 'italic'
            elif k == '{tts}fontWeight':
                style['bold'] = v == 'bold'
            elif k == '{tts}textDecoration':
                style['underline'] = v == 'underline'
        return style

    def __process_time(self, text):
        coefs = [3600, 60, 1]
        time = 0.0

        offset_match = re.match(r'(\d+)(:?\.\d+)?(h|m|s|ms|f|t)', text)
        if offset_match:
            return float(offset_match.group(1)) * {
                'h': 3600.0,
                'm': 60.0,
                's': 1.0,
                'ms': 0.001,
                'f': 1.0/(self.frameRate * self.frameRateMultiplier),
                't': 1.0/self.tickRate
            }.get(offset_match.group(2), 1.0)
        params = text.split(':')
        if len(params) in (3, 4):
            if len(params) == 4:
                frames = params[3].split('.', 2)
                if len(frames) == 1:
                    params[2] = float(params[2]) + float(params[3]) / (self.frameRate * self.frameRateMultiplier)
                else:
                    params[2] = float(params[2]) + (
                        float(frames[0]) / self.frameRate +
                        float(frames[1]) / (self.frameRate * self.subFrameRate)
                    ) * self.frameRateMultiplier
                del params[3]
            for c, v in zip(coefs, params):
                time += c*float(v)
            return time
        return 0.0

    def _parseXML(self):
        # Normalize namespaces to a single alias. The draft namespace are still used in some file which makes searching for tags cumbersome
        namespace_clean = {
            'http://www.w3.org/2006/10/ttaf1': 'tt',
            'http://www.w3.org/2006/04/ttaf1': 'tt',
            'http://www.w3.org/ns/ttml': 'tt',
            'http://www.w3.org/2006/10/ttaf1#styling': 'tts',
            'http://www.w3.org/2006/04/ttaf1#styling': 'tts',
            'http://www.w3.org/ns/ttml#styling': 'tts',
            'http://www.w3.org/2006/10/ttaf1#parameter': 'ttp',
            'http://www.w3.org/2006/04/ttaf1#parameter': 'ttp',
            'http://www.w3.org/ns/ttml#parameter': 'ttp',
        }
        def normalize_qname(name):
            if name[0] == '{':
                (ns, name) = name[1:].split('}', 1)
                ns = namespace_clean.get(ns, ns)
                return '{%s}%s' % (ns, name)
            return name

        xml = ElementTree.parse(self.source)
        for element in xml.getiterator():
            element.tag = normalize_qname(element.tag)
            for k, v in list(element.items()):
                new_k = normalize_qname(k)
                if k != new_k:
                    del element.attrib[k]
                    element.attrib[new_k] = v

        # Define style aliases
        styles = {}
        regions = {}

        root = xml.getiterator()[0]
        if int(root.get('{ttp}tickRate', 0)) > 0:
            self.tickRate = int(root.get('{ttp}tickRate'))
        if int(root.get('{ttp}frameRate', 0)) > 0:
            self.frameRate = int(root.get('{ttp}frameRate'))
        if int(root.get('{ttp}subFrameRate', 0)) > 0:
            self.subFrameRate = int(root.get('{ttp}subFrameRate'))
        if root.get('{ttp}frameRateMultiplier'):
            num, denom = root.get('{ttp}frameRateMultiplier').split(' ')
            self.frameRateMultiplier = float(num) / float(denom)
        if not self.tickRate:
            self.tickRate = self.frameRate * self.subFrameRate * self.frameRateMultiplier

        # Build a cache for the default styles
        for style_tag in xml.findall('{tt}head/{tt}styling/{tt}style'):
            style = self.__parse_style(style_tag)
            styles[style_tag.get('{http://www.w3.org/XML/1998/namespace}id')] = style

        # Build a cache for the default style of the regions
        for region_tag in xml.findall('{tt}head/{tt}layout/{tt}region'):
            region = self.__parse_style(region_tag)
            regions[region_tag.get('{http://www.w3.org/XML/1998/namespace}id')] = region

        def compute_style_tree(element):
            style_ref = element.get('style')
            region_ref = element.get('region')

            style = {}
            if region_ref:
                style.update(regions[region_ref])
            if style_ref:
                style.update(styles[style_ref])
            style.update(self.__parse_style(element))

            return style

        def styleToHtml(tag, value):
            return {
                'bold': ('b', '<b>'),
                'italic': ('i', '<i>'),
                'underline': ('u', '<u>'),
                'color': ('font', '<font color="%s">' % value),
            }[tag]

        def openTags(output, style_pairs):
            (before, after) = style_pairs
            for tag in sorted(after.keys()):
                new_value = after[tag]
                old_value = before.get(tag, None)
                if old_value == None and new_value:
                    html = styleToHtml(tag, new_value)
                    output.openTag(html[0], html[1])
                elif old_value != new_value:
                    if new_value:
                        html = styleToHtml(tag, new_value)
                        output.openTag(html[0], html[1])
                    else:
                        output.closeTag(styleToHtml(tag, new_value)[0])

        def closeTags(output, style_pairs):
            (before, after) = style_pairs
            for tag in sorted(list(after.keys()), reverse=True):
                new_value = after[tag]
                old_value = before.get(tag, None)
                if old_value == None and new_value:
                    output.closeTag(styleToHtml(tag, new_value)[0])
                elif old_value != new_value:
                    if new_value:
                        output.closeTag(styleToHtml(tag, new_value)[0])
                    else:
                        html = styleToHtml(tag, before[tag])
                        output.openTag(html[0], html[1])

        # Store the subs in a list
        self.subs = []
        prev_sub = None
        content = None
        sub_grouping = False
        for sub in xml.findall('{tt}body/{tt}div/{tt}p'):
            begin = self.__process_time(sub.get('begin'))
            if not sub.get('end'):
                end = begin + self.__process_time(sub.get('dur'))
            else:
                end = self.__process_time(sub.get('end'))

            style_stack = [{'color': '#ffffff'}] # default color

            if not prev_sub or begin != prev_sub[0] or end != prev_sub[1]:
                content = RichText(self.richFormatting)
                sub_grouping = False
            else:
                content.write("\n")
                sub_grouping = True

            def parseChildTree(element_list):
                for child in element_list:
                    style_stack.append(compute_style_tree(child))
                    openTags(content, style_stack[-2:])
                    if child.text and child.text.strip():
                        content.write(child.text.strip())
                    if child.tag == '{tt}br':
                        content.write("\n")
                    parseChildTree(child.getchildren())
                    if child.tail and child.tail.strip():
                        content.write(child.tail.strip())
                    closeTags(content, style_stack[-2:])
                    style_stack.pop()

            parseChildTree([sub])

            # try to regroup subtitles if possible
            if sub_grouping:
                self.subs[-1][2] = str(content)
            else:
                prev_sub = [begin, end, str(content)]
                self.subs.append(prev_sub)

    def __iter__(self):
        return iter(self.subs)
