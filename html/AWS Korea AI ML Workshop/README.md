# workshop-sample

This project allows you to scaffold a workshop using a AWS-styled Hugo theme similar to those available at [cdkworkshop.com](https://cdkworkshop.com/), [eksworkshop.com](https://eksworkshop.com/), or [ecsworkshop.com](https://ecsworkshop.com/).

```bash
.
├── metadata.yml                      <-- Metadata file with descriptive information about the workshop
├── README.md                         <-- This instructions file
├── deck                              <-- Directory for presentation deck
├── resources                         <-- Directory for workshop resources
│   ├── code                          <-- Directory for workshop modules code
│   ├── policies                      <-- Directory for workshop modules IAM Roles and Policies
│   └── templates                     <-- Directory for workshop modules CloudFormation templates
└── workshop                          
    ├── buildspec.yml                 <-- AWS CodeBuild build script for building the workshop website
    ├── config.toml                   <-- Hugo configuration file for the workshop website
    └── content                       <-- Markdown files for pages/steps in workshop
    └── static                        <-- Any static assets to be hosted alongside the workshop (ie. images, scripts, documents, etc)
    └── themes                        <-- AWS Style Hugo Theme (Do not edit!)
```

## Requirements

1. [Clone this repository](https://github.com/aws-samples/aws-ai-ml-workshop-kr).
2. [Install Hugo locally](https://gohugo.io/overview/quickstart/).

## Create your project

If you haven't already, make a copy of this entire directory and rename it something descriptive, similar to the title of your workshop.

```bash
cp -R Aws-workshop-template/ my-first-worshop/
```

## What's Included

This project the following folders:

* `deck`: The location to store your presentation materials, if not already stored centrally in a system like KnowledgeMine or Wisdom.
* `resources`: Store any example code, IAM policies, or Cloudformation templates needed by your workshop here.
* `workshop`: This is the core workshop folder. This is generated as HTML and hosted for presentation for customers.


## Navigate to the `workshop` directory

All command line directions in this documentation assume you are in the `workshop` directory. Navigate there now, if you aren't there already.

```bash
cd my-first-workshop/workshop
```

## Create your first chapter page

Chapters are pages that contain other child pages. It has a special layout style and usually just contains a _brief abstract_ of the section.

```markdown
Discover what this template is all about and the core concepts behind it.
```

This template provides archetypes to create skeletons for your workshop. Begin by creating your first chapter page with the following command

```bash
cd workshop
hugo new --kind chapter intro/_index.en.md
```

By opening the given file, you should see the property `chapter=true` on top, meaning this page is a _chapter_.

By default all chapters and pages are created as a draft. If you want to render these pages, remove the property `draft = true` from the metadata.

## Create your first content pages

Then, create content pages inside the previously created chapter. Here are two ways to create content in the chapter:

```bash
hugo new intro/first-content.en.md
hugo new intro/second-content/_index.en.md
```

Feel free to edit thoses files by adding some sample content and replacing the `title` value in the beginning of the files. 

## Launching the website locally

Launch by using the following command:

```bash
hugo serve
```

Go to `http://localhost:1313`

You should notice three things:

1. You have a left-side **Intro** menu, containing two submenus with names equal to the `title` properties in the previously created files.
2. The home page explains how to customize it by following the instructions.
3. When you run `hugo serve`, when the contents of the files change, the page automatically refreshes with the changes. Neat!

Alternatively, you can run the following command in a terminal window to tell Hugo to automatically rebuild whenever a file is changed. This can be helpful when rapidly iterating over content changes.

```bash
hugo serve -D
```