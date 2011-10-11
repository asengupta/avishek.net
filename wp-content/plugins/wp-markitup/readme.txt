=== WP MarkItUp! ===
Contributors: stefano.verna
Donate link: http://wpmarkitup.stefanoverna.com/
Tags: markitup, editor, tinymce, fckeditor, markdown, textile, wiki, wikimedia, syntax, xhtml, toolbar, skins
Requires at least: 2.7
Tested up to: 2.7
Stable tag: trunk

Replaces the old "quicktags" toolbar with MarkItUp, a lightweight costumizable markup editor with XHTML, Textile, Wiki, Markdown and BBcode toolbars!

== Description ==

WP MarkItUp! is the Wordpress plugin that replaces the old "quicktags" toolbar with [MarkItUp!](http://markitup.jaysalvat.com/home/ "MarkItUp! official website"), a lightweight jQuery plugin that allows to turn any textarea into an highly customizable markup editor. **XHTML, Textile, Wiki, Markdown and BBcode** toolbars are already provided out-of-the-box, but even your own markup syntax can be easily implemented with this system. The plugin currently features three different skins for the toolbar, new ones are about to come.

== Installation ==

1. Download the zipped plugin (the latest version will always be available at the [WP MarkItUp! official page](http://wpmarkitup.stefanoverna.com/ "WP MarkItUp!")).
2. Unzip it and upload the whole `wp_markitup` directory to your `/wp-content/plugins/` directory.
3. Make sure the permissions on all .php files it are at least 755. If you can SSH, simply go to  the `wp_markitup` directory and type `find . -name '*.php' | xargs chmod 755`. Alternatively, just use any FTP client (Filezilla, Cyberduck) to do it manually.
4. Activate the plugin through the **Plugins** menu in WordPress.
5. Go to the WP MarkItUp! setting page under the **Settings** section. Here you can choose the toolbar and the skin to activate from the available ones.
6. There's no step six. Enjoy your brand new 21st century editor ;)

== Frequently Asked Questions ==

= I have successfully deactivated the plugin, but the visual tinyMCE editor doesn't show up! =

You need to go to your Profile page, and check the "Use the visual editor when writing" option.

= How can I add my own personal tagset to the plugin? =

You basically need to create a new directory inside the `/wp-content/plugins/wp_markitup/markitup/sets/` directory containing the toolbar stylesheet `style.css`, `set.php` which is the markItUp! toolbar specification and `infos.php`, a simple file used by the plugin to know if the tagset needs some external plugin to work.  
Please note that `set.php`, despite its extension, it's a common Javascript source code file: the php it's just for localization purposes.
 
Your're strongly suggested to read the well written [markItUp! documentation](http://markitup.jaysalvat.com/documentation/ "MarkItUp! documentation") on how to setup your custom tagset, and watch a couple of [markItUp! examples](http://markitup.jaysalvat.com/examples/) for yourself.

= How can I add a new skin/edit an existing one? =

Even easier. You just need to have some basic CSS knowledge and have to look at the `wp_markitup/markitup/skins/` directory. You'll find a directory for each existing skin. Each dir will contain a stylesheet named `style.css`. It's common to see also an `images` directory used by the stylesheet itself.

= I would like to become a WP MarkItUp! translator! =

Great! You can start installing [Poedit](http://www.poedit.net/download.php) and opening the `wp_markitup/languages/markitup-it_IT.po` file with it. It's all pretty straightforward. If you have problems or you want to submit me the finished translation, don't hesitate to [contact me](http://www.stefanoverna.com/about/).

= Feature Request/Bug Reporting =

If you find any problems/bugs/annoyances or if you feel like something really important is missing, than please let me know commenting at the [WP MarkItUp! official page](http://wpmarkitup.stefanoverna.com/ "WP MarkItUp!"). Thanks for the support!

== Screenshots ==

1. WP MarkItUp! doing his work with MarkDown syntax toolbar enabled.
2. WP MarkItUp! with another skin activated. The toolbar is now for XHTML tags.
3. WP MarkItUp! settings page showing the different toolbars available.
4. WP MarkItUp! settings page showing the different skins available.
5. WP MarkItUp! settings page showing a warning if you're enabling a toolbar that needs an external plugin to work properly.
6. WP MarkItUp! showing a warning in the "Write a Post" page if you have enabled a toolbar that needs an external plugin to work properly.

