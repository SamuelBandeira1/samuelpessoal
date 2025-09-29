# Samuel Pessoal

Este projeto contém uma página estática simples para divulgar informações pessoais ou profissionais. As imagens utilizadas são vetores
mínimos criados como placeholders, o que facilita a visualização dos diffs e a customização.

## Estrutura

```
assets/
└── img/
    ├── hero-placeholder.svg
    ├── profile-placeholder.svg
    └── project-placeholder.svg
index.html
README.md
```

### Imagens vetoriais

Todas as imagens em `assets/img/` estão no formato SVG. Elas foram desenhadas com gradientes suaves e texto explicando o conteúdo que deve ser
substituído. Como são vetoriais, você pode editá-las diretamente em um editor de texto ou em ferramentas como Figma, Illustrator e Inkscape
antes de exportar suas versões definitivas.

Para atualizar uma imagem:

1. Substitua o arquivo `.svg` correspondente por outro com o mesmo nome.
2. Caso prefira usar um formato raster (PNG, JPG, WEBP), mantenha a mesma nomenclatura mas atualize o caminho em `index.html` para o novo arquivo.
3. Ajuste os atributos `width`, `height` e `alt` da tag `<img>` conforme necessário para refletir o novo conteúdo.

### Desenvolvimento

Nenhuma ferramenta específica é necessária: basta abrir `index.html` em um navegador para visualizar o resultado. Se desejar publicar a página, é
possível fazer o deploy em serviços estáticos como GitHub Pages, Netlify ou Vercel.
