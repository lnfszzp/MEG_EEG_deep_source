// Only builds the explanation document; never imports or changes experiment code.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const d = require('docx');
const input = path.join(__dirname, '当前算法说明内容.json');
const content = JSON.parse(fs.readFileSync(input, 'utf8'));
const root = path.dirname(__dirname);
const children = [];
let equationCount = 0;

// The installed docx package creates native, editable Word equations.
function math(value) {
    if (typeof value === 'string') return [new d.MathRun(value)];
    if (Array.isArray(value)) return value.flatMap(math);
    if (value.sub) return [new d.MathSubScript({ children: math(value.sub[0]), subScript: math(value.sub[1]) })];
    if (value.sup) return [new d.MathSuperScript({ children: math(value.sup[0]), superScript: math(value.sup[1]) })];
    if (value.frac) return [new d.MathFraction({ numerator: math(value.frac[0]), denominator: math(value.frac[1]) })];
    if (value.sum) return [new d.MathSum({ children: math(value.sum[1]), subScript: math(value.sum[0]) })];
    if (value.sqrt) return [new d.MathRadical({ children: math(value.sqrt) })];
    if (value.norm) {
        const children = [new d.MathRun('∥'), ...math(value.norm[0]), new d.MathRun('∥')];
        return value.norm.length === 3
            ? [new d.MathSubSuperScript({ children, subScript: math(value.norm[1]), superScript: math(value.norm[2]) })]
            : [new d.MathSubScript({ children, subScript: math(value.norm[1]) })];
    }
    throw new Error(`Unknown equation node ${JSON.stringify(value)}`);
}
function paragraph(text, options = {}) {
    assert.equal(typeof text, 'string');
    return new d.Paragraph({ children: [new d.TextRun({ text })], ...options });
}
const line = { style: d.BorderStyle.SINGLE, size: 4, color: 'D9D9D9' };
const records = [];
for (const [index, section] of content.sections.entries()) {
    assert.ok(section.title && section.blocks.length);
    if (!index) {
        children.push(paragraph(content.title, { style: 'Title' }));
        children.push(paragraph(content.subtitle, { style: 'Subtitle' }));
    }
    children.push(paragraph(section.title, { heading: d.HeadingLevel.HEADING_1, pageBreakBefore: index > 0 }));
    for (const block of section.blocks) {
        if (block.text) children.push(paragraph(block.text));
        else if (block.heading) children.push(paragraph(block.heading, { heading: d.HeadingLevel.HEADING_2 }));
        else if (block.equation) {
            equationCount++;
            children.push(new d.Paragraph({
                children: [new d.Math({ children: math(block.equation) })],
                alignment: d.AlignmentType.CENTER, keepLines: true,
                spacing: { before: 90, after: 90 },
            }));
        } else if (block.table) {
            const table = block.table;
            assert.equal(table.widths.reduce((a, b) => a + b, 0), 9354);
            children.push(new d.Table({
                width: { size: 9354, type: d.WidthType.DXA },
                columnWidths: table.widths,
                layout: d.TableLayoutType.FIXED,
                borders: { top: line, bottom: line, left: line, right: line, insideHorizontal: line, insideVertical: line },
                rows: [table.headers, ...table.rows].map((row, ri) => {
                    assert.equal(row.length, table.widths.length);
                    return new d.TableRow({ tableHeader: ri === 0, cantSplit: true, children: row.map((text, ci) => new d.TableCell({
                        width: { size: table.widths[ci], type: d.WidthType.DXA },
                        verticalAlign: d.VerticalAlign.CENTER,
                        margins: { top: 75, bottom: 75, left: 110, right: 110 },
                        shading: { fill: ri === 0 ? 'DCE6F1' : 'FFFFFF' },
                        children: [new d.Paragraph({
                            children: [new d.TextRun({ text, size: 21, bold: ri === 0, color: '000000' })],
                            alignment: ci === 0 ? d.AlignmentType.LEFT : (table.centerColumns || []).includes(ci) ? d.AlignmentType.CENTER : d.AlignmentType.LEFT,
                            spacing: { after: 0, line: 285 },
                        })],
                    })) });
                }),
            }));
            children.push(new d.Paragraph({ spacing: { after: 55 } }));
        } else throw new Error(`Unknown block ${JSON.stringify(block)}`);
    }
}
assert.ok(equationCount >= 25);
assert.equal(content.sections.length, 19);
for (const relative of content.source_files) {
    const payload = fs.readFileSync(path.join(root, relative));
    records.push({ path: relative, sha256: crypto.createHash('sha256').update(payload).digest('hex') });
}
const document = new d.Document({
    creator: 'OASTER algorithm documentation',
    title: content.title,
    description: 'Current implemented ERP inverse and conditional deep-presence gate, code snapshot 2026-10-10',
    styles: {
        default: { document: { run: { font: { ascii: 'Cambria', hAnsi: 'Cambria', eastAsia: '宋体' }, size: 24, color: '000000' }, paragraph: { spacing: { after: 115, line: 315 }, widowControl: true } } },
        paragraphStyles: [
            { id: 'Title', name: 'Title', basedOn: 'Normal', run: { font: '微软雅黑', size: 38, bold: true, color: '000000' }, paragraph: { spacing: { after: 140 } } },
            { id: 'Subtitle', name: 'Subtitle', basedOn: 'Normal', run: { size: 21, color: '000000' }, paragraph: { spacing: { after: 220 } } },
            { id: 'Heading1', name: 'heading 1', basedOn: 'Normal', next: 'Normal', run: { font: '微软雅黑', size: 30, bold: true, color: '000000' }, paragraph: { keepNext: true, spacing: { before: 70, after: 175 } } },
            { id: 'Heading2', name: 'heading 2', basedOn: 'Normal', next: 'Normal', run: { font: '微软雅黑', size: 25, bold: true, color: '000000' }, paragraph: { keepNext: true, spacing: { before: 160, after: 100 } } },
        ],
    },
    sections: [{
        properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1050, bottom: 1100, left: 1276, right: 1276, footer: 500 } } },
        footers: { default: new d.Footer({ children: [new d.Paragraph({ alignment: d.AlignmentType.CENTER, children: [new d.TextRun({ children: [d.PageNumber.CURRENT], size: 18, color: '000000' })] })] }) },
        children,
    }],
});
const output = path.join(__dirname, 'OASTER当前ERP算法完整说明20261010.docx');
d.Packer.toBuffer(document).then(buffer => {
    fs.writeFileSync(output, buffer);
    fs.writeFileSync(path.join(__dirname, '当前算法说明核对清单.json'), JSON.stringify({
        code_commit: content.code_commit, equation_count: equationCount,
        source_files: records,
        content_sha256: crypto.createHash('sha256').update(fs.readFileSync(input)).digest('hex'),
        output_sha256: crypto.createHash('sha256').update(buffer).digest('hex'),
        experiment_code_modified: false,
    }, null, 2) + '\n');
    console.log(JSON.stringify({ output, equationCount, sections: content.sections.length }));
}).catch(error => { console.error(error); process.exitCode = 1; });
